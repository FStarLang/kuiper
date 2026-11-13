#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <cmath>
#include <vector>

__device__ float add_ftz(float a, float b) {
  float c;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

__device__ float mul_ftz(float a, float b) {
  float c;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

__device__ float fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ float rsqrt_ftz(float a) {
  float b;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ float rcp_ftz(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ float exp2_ftz(float a) {
  float b;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ uint32_t pack_bf16(at::BFloat16 a, at::BFloat16 b) {
  return uint32_t(a.x) | (uint32_t(b.x) << 16);
}

__global__ void embed_rope(const at::BFloat16* weights, const int64_t* prompt,
                          at::BFloat16* hidden, at::BFloat16* rope) {
  for (int row = 0; row < 8; ++row) {
    int64_t token = row < 7 ? prompt[row] : 0;
    for (int col = 0; col < 1024; ++col)
      hidden[row * 1024 + col] = weights[token * 1024 + col];
  }
  for (int col = 0; col < 64; ++col) {
    float inv = 1.0f / powf(1000000.0f, float(col) / 64.0f);
    for (int row = 0; row < 4096; ++row) {
      float angle = inv * float(row);
      at::BFloat16 c(cosf(angle)), s(sinf(angle));
      rope[row * 128 + col] = c;
      rope[row * 128 + col + 64] = c;
      rope[4096 * 128 + row * 128 + col] = s;
      rope[4096 * 128 + row * 128 + col + 64] = s;
    }
  }
}

__global__ void rms_norm(const at::BFloat16* input, const at::BFloat16* weights,
                         at::BFloat16* output, int64_t weight_offset, int rows) {
  float partial[256];
  float warp_sum[8];
  for (int row = 0; row < rows; ++row) {
    for (int thread = 0; thread < 256; ++thread) {
      float sum = 0.0f;
      for (int i = thread; i < 1024; i += 256) {
        float v = float(input[row * 1024 + i]);
        sum = fma_ftz(v, v, sum);
      }
      partial[thread] = sum;
    }
    for (int warp = 0; warp < 8; ++warp) {
      for (int offset = 16; offset; offset /= 2)
        for (int lane = 0; lane < offset; ++lane)
          partial[warp * 32 + lane] =
              add_ftz(partial[warp * 32 + lane], partial[warp * 32 + lane + offset]);
      warp_sum[warp] = partial[warp * 32];
    }
    for (int offset = 4; offset; offset /= 2)
      for (int lane = 0; lane < offset; ++lane)
        warp_sum[lane] = add_ftz(warp_sum[lane], warp_sum[lane + offset]);
    float scale = rsqrt_ftz(fma_ftz(warp_sum[0], 1.0f / 1024.0f, 1.0e-6f));
    for (int i = 0; i < 1024; ++i)
      output[row * 1024 + i] = at::BFloat16(
          mul_ftz(float(input[row * 1024 + i]), mul_ftz(scale, float(weights[weight_offset + i]))));
  }
}

__global__ void linear_mma(const at::BFloat16* input, const at::BFloat16* weights,
                           const at::BFloat16* residual, at::BFloat16* output,
                           int64_t weight_offset, int n, int k, int rows,
                           int layout, int add_residual) {
  __shared__ float tile[16 * 8];
  int lane = threadIdx.x;
  int group = lane / 4;
  int pair = (lane % 4) * 2;
  for (int block = 0; block < n; block += 16) {
    int wr0 = block + group;
    int wr1 = wr0 + 8;
    if (layout == 1) {
      int subgroup = wr0 / 512;
      int inner = wr0 % 512;
      wr0 = inner < 256 ? subgroup * 256 + inner
          : (inner < 384 ? 2048 + subgroup * 128 + inner - 256
                         : 3072 + subgroup * 128 + inner - 384);
      wr1 = wr0 + 8;
    }
    if (layout == 2) {
      int inner = wr0 % 128;
      wr0 = wr0 / 128 * 64 + inner % 64 + (inner >= 64 ? 3072 : 0);
      wr1 = wr0 + 8;
    }
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int base = 0; base < k; base += 16) {
      int col = base + pair;
      at::BFloat16 a[8];
      for (int i = 0; i < 8; ++i) {
        int r = (i / 2) % 2 ? wr1 : wr0;
        int c = col + i % 2 + (i / 4) * 8;
        a[i] = layout == 3 && r >= 151936 ? at::BFloat16(0.0f)
                                         : weights[weight_offset + int64_t(r) * k + c];
      }
      uint32_t a0 = pack_bf16(a[0], a[1]);
      uint32_t a1 = pack_bf16(a[2], a[3]);
      uint32_t a2 = pack_bf16(a[4], a[5]);
      uint32_t a3 = pack_bf16(a[6], a[7]);
      uint32_t b0 = group < rows
          ? pack_bf16(input[group * k + col], input[group * k + col + 1]) : 0;
      uint32_t b1 = group < rows
          ? pack_bf16(input[group * k + col + 8], input[group * k + col + 9]) : 0;
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    tile[group * 8 + pair] = d0;
    tile[group * 8 + pair + 1] = d1;
    tile[(group + 8) * 8 + pair] = d2;
    tile[(group + 8) * 8 + pair + 1] = d3;
    __syncwarp();
    if (lane == 0) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < 16; ++c) {
          at::BFloat16 v(tile[c * 8 + r]);
          if (add_residual)
            v = at::BFloat16(__hadd(__ushort_as_bfloat16(v.x),
                                    __ushort_as_bfloat16(residual[r * n + block + c].x)));
          output[r * n + block + c] = v;
        }
      }
    }
    __syncwarp();
  }
}

__global__ void norm_rope(const at::BFloat16* input, const at::BFloat16* weights,
                          const at::BFloat16* rope, at::BFloat16* output,
                          int64_t weight_offset) {
  float partial[128];
  float warp_sum[32];
  at::BFloat16 normalized[128];
  for (int i = 0; i < 8 * 4096; ++i)
    output[i] = input[i];
  for (int row = 0; row < 7; ++row) {
    for (int group = 0; group < 8; ++group) {
      for (int head = 0; head < 3; ++head) {
        int base = row * 4096 + group * 512 + head * 128;
        for (int i = 0; i < 128; ++i) {
          float v = float(input[base + i]);
          partial[i] = fma_ftz(v, v, 0.0f);
        }
        for (int warp = 0; warp < 4; ++warp) {
          for (int offset = 16; offset; offset /= 2)
            for (int lane = 0; lane < offset; ++lane)
              partial[warp * 32 + lane] =
                  add_ftz(partial[warp * 32 + lane], partial[warp * 32 + lane + offset]);
          warp_sum[warp] = partial[warp * 32];
        }
        for (int i = 4; i < 32; ++i)
          warp_sum[i] = 0.0f;
        for (int offset = 16; offset; offset /= 2)
          for (int lane = 0; lane < offset; ++lane)
            warp_sum[lane] = add_ftz(warp_sum[lane], warp_sum[lane + offset]);
        float scale = rsqrt_ftz(fma_ftz(warp_sum[0], 1.0f / 128.0f, 1.0e-6f));
        for (int i = 0; i < 128; ++i) {
          float weight = float(weights[weight_offset + (head == 2 ? 128 : 0) + i]);
          normalized[i] = at::BFloat16(mul_ftz(float(input[base + i]), mul_ftz(scale, weight)));
        }
        for (int i = 0; i < 128; ++i) {
          float c = float(rope[row * 128 + i]);
          float s = float(rope[4096 * 128 + row * 128 + i]);
          float other = float(normalized[(i + 64) % 128]);
          output[base + i] = at::BFloat16(fma_ftz(
              float(normalized[i]), c, mul_ftz(i < 64 ? -other : other, s)));
        }
      }
    }
  }
}

__global__ void attention_mma(const at::BFloat16* input, at::BFloat16* output) {
  __shared__ float tile[16 * 8];
  __shared__ at::BFloat16 probability[16 * 64];
  __shared__ float denominator[16];
  int lane = threadIdx.x;
  int group = lane / 4;
  int pair = (lane % 4) * 2;
  for (int head = 0; head < 8; ++head) {
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int base = 0; base < 128; base += 16) {
      at::BFloat16 a[8], b[4];
      for (int i = 0; i < 8; ++i) {
        int row = group + ((i / 2) % 2) * 8;
        int col = base + pair + i % 2 + (i / 4) * 8;
        a[i] = row < 14
            ? input[(row / 2) * 4096 + head * 512 + (row % 2) * 128 + col]
            : at::BFloat16(0.0f);
      }
      for (int i = 0; i < 4; ++i)
        b[i] = group < 7
            ? input[group * 4096 + head * 512 + 256 + base + pair + i % 2 + (i / 2) * 8]
            : at::BFloat16(0.0f);
      uint32_t a0 = pack_bf16(a[0], a[1]), a1 = pack_bf16(a[2], a[3]);
      uint32_t a2 = pack_bf16(a[4], a[5]), a3 = pack_bf16(a[6], a[7]);
      uint32_t b0 = pack_bf16(b[0], b[1]), b1 = pack_bf16(b[2], b[3]);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    tile[group * 8 + pair] = d0;
    tile[group * 8 + pair + 1] = d1;
    tile[(group + 8) * 8 + pair] = d2;
    tile[(group + 8) * 8 + pair + 1] = d3;
    __syncwarp();
    if (lane == 0) {
      for (int row = 0; row < 16; ++row) {
        float maxima[4], sums[4];
        for (int p = 0; p < 4; ++p) {
          maxima[p] = -CUDART_INF_F;
          sums[p] = 0.0f;
          for (int j = 0; j < 16; ++j) {
            int col = p * 2 + (j / 2) * 8 + j % 2;
            float v = row < 14 && col <= row / 2 ? tile[row * 8 + col] : -CUDART_INF_F;
            maxima[p] = fmaxf(maxima[p], v);
          }
        }
        float maximum = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
        float scaled_maximum = mul_ftz(maximum, 0.12751743197441101074f);
        for (int p = 0; p < 4; ++p) {
          for (int j = 0; j < 16; ++j) {
            int col = p * 2 + (j / 2) * 8 + j % 2;
            float v = row < 14 && col <= row / 2 ? tile[row * 8 + col] : -CUDART_INF_F;
            float e = v != -CUDART_INF_F
                ? exp2_ftz(fma_ftz(v, 0.12751743197441101074f, -scaled_maximum)) : 0.0f;
            probability[row * 64 + col] = at::BFloat16(e);
            sums[p] = add_ftz(sums[p], e);
          }
        }
        float rescale = maximum == -CUDART_INF_F ? 1.0f
            : exp2_ftz(fma_ftz(-CUDART_INF_F, 0.12751743197441101074f, -scaled_maximum));
        denominator[row] = add_ftz(rescale, add_ftz(add_ftz(sums[0], sums[1]),
                                                   add_ftz(sums[2], sums[3])));
      }
    }
    __syncwarp();
    for (int block = 0; block < 128; block += 8) {
      d0 = 0.0f;
      d1 = 0.0f;
      d2 = 0.0f;
      d3 = 0.0f;
      for (int base = 0; base < 64; base += 16) {
        int col = base + pair;
        uint32_t a0 = pack_bf16(probability[group * 64 + col], probability[group * 64 + col + 1]);
        uint32_t a1 = pack_bf16(probability[(group + 8) * 64 + col],
                                probability[(group + 8) * 64 + col + 1]);
        uint32_t a2 = pack_bf16(probability[group * 64 + col + 8], probability[group * 64 + col + 9]);
        uint32_t a3 = pack_bf16(probability[(group + 8) * 64 + col + 8],
                                probability[(group + 8) * 64 + col + 9]);
        at::BFloat16 b[4];
        for (int i = 0; i < 4; ++i) {
          int row = col + i % 2 + (i / 2) * 8;
          b[i] = row < 7 ? input[row * 4096 + head * 512 + 384 + block + group]
                         : at::BFloat16(0.0f);
        }
        uint32_t b0 = pack_bf16(b[0], b[1]), b1 = pack_bf16(b[2], b[3]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
      tile[group * 8 + pair] = d0;
      tile[group * 8 + pair + 1] = d1;
      tile[(group + 8) * 8 + pair] = d2;
      tile[(group + 8) * 8 + pair + 1] = d3;
      __syncwarp();
      if (lane == 0) {
        for (int row = 0; row < 14; ++row)
          for (int col = 0; col < 8; ++col)
            output[(row / 2) * 2048 + head * 256 + (row % 2) * 128 + block + col] =
                at::BFloat16(mul_ftz(tile[row * 8 + col], rcp_ftz(denominator[row])));
      }
      __syncwarp();
    }
  }
  if (lane == 0)
    for (int col = 0; col < 2048; ++col)
      output[7 * 2048 + col] = at::BFloat16(0.0f);
}

__global__ void silu_mul(const at::BFloat16* input, at::BFloat16* output) {
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 3072; ++col) {
      int index = row * 6144 + (col / 64) * 128 + col % 64;
      float v = float(input[index]);
      float e = exp2_ftz(mul_ftz(-v, 1.44269504088896340736f));
      at::BFloat16 s(mul_ftz(v, rcp_ftz(add_ftz(1.0f, e))));
      output[row * 3072 + col] = at::BFloat16(
          __hmul(__ushort_as_bfloat16(s.x), __ushort_as_bfloat16(input[index + 64].x)));
    }
  }

}

__global__ void select_token(const at::BFloat16* logits, const int64_t* prompt,
                             int64_t* output) {
    float value[256], chunk_value[96];
    int64_t index[256], chunk_index[96];
    float initial = -CUDART_INF_F;
    for (int chunk = 0; chunk < 96; ++chunk) {
      for (int thread = 0; thread < 128; ++thread) {
        value[thread] = initial;
        index[thread] = -1;
        for (int i = thread; i < 1600 && chunk * 1600 + i < 151936; i += 128) {
          float v = float(logits[6 * 153600 + chunk * 1600 + i]);
          if (v > value[thread]) {
            value[thread] = v;
            index[thread] = i;
          }
        }
      }
      for (int warp = 0; warp < 4; ++warp) {
        for (int offset = 16; offset; offset /= 2) {
          for (int lane = 0; lane < offset; ++lane) {
            int i = warp * 32 + lane;
            if (value[i + offset] > value[i]) {
              value[i] = value[i + offset];
              index[i] = index[i + offset];
            }
          }
        }
        value[warp] = value[warp * 32];
        index[warp] = index[warp * 32];
      }
      for (int offset = 2; offset; offset /= 2)
        for (int lane = 0; lane < offset; ++lane)
          if (value[lane + offset] > value[lane]) {
            value[lane] = value[lane + offset];
            index[lane] = index[lane + offset];
          }
      chunk_value[chunk] = value[0];
      chunk_index[chunk] = index[0];
    }
    for (int thread = 0; thread < 256; ++thread) {
      value[thread] = initial;
      index[thread] = -1;
      if (thread < 96 && chunk_value[thread] > initial) {
        value[thread] = chunk_value[thread];
        index[thread] = (int64_t(thread) << 32) | chunk_index[thread];
      }
    }
    for (int warp = 0; warp < 8; ++warp) {
      for (int offset = 16; offset; offset /= 2)
        for (int lane = 0; lane < offset; ++lane) {
          int i = warp * 32 + lane;
          if (value[i + offset] > value[i]) {
            value[i] = value[i + offset];
            index[i] = index[i + offset];
          }
        }
      value[warp] = value[warp * 32];
      index[warp] = index[warp * 32];
    }
    for (int offset = 4; offset; offset /= 2)
      for (int lane = 0; lane < offset; ++lane)
        if (value[lane + offset] > value[lane]) {
          value[lane] = value[lane + offset];
          index[lane] = index[lane + offset];
        }
    for (int i = 0; i < 7; ++i)
      output[i] = prompt[i];
    output[7] = index[0] == -1 ? -1 : (index[0] >> 32) * 1600 + (index[0] & 0xffffffffLL);
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
  auto embedding = torch::empty({8, 1024}, input0.options());
  auto rope = torch::empty({2, 4096, 128}, input0.options());
  auto normalized_0 = torch::empty({8, 1024}, input0.options());
  auto qkv_0 = torch::empty({8, 4096}, input0.options());
  auto rotated_0 = torch::empty({8, 4096}, input0.options());
  auto attention_0 = torch::empty({8, 2048}, input0.options());
  auto residual_0 = torch::empty({8, 1024}, input0.options());
  auto postnorm_0 = torch::empty({8, 1024}, input0.options());
  auto gate_up_0 = torch::empty({8, 6144}, input0.options());
  auto activated_0 = torch::empty({8, 3072}, input0.options());
  auto hidden_0 = torch::empty({8, 1024}, input0.options());
  auto normalized_1 = torch::empty({8, 1024}, input0.options());
  auto qkv_1 = torch::empty({8, 4096}, input0.options());
  auto rotated_1 = torch::empty({8, 4096}, input0.options());
  auto attention_1 = torch::empty({8, 2048}, input0.options());
  auto residual_1 = torch::empty({8, 1024}, input0.options());
  auto postnorm_1 = torch::empty({8, 1024}, input0.options());
  auto gate_up_1 = torch::empty({8, 6144}, input0.options());
  auto activated_1 = torch::empty({8, 3072}, input0.options());
  auto hidden_1 = torch::empty({8, 1024}, input0.options());
  auto normalized_2 = torch::empty({8, 1024}, input0.options());
  auto qkv_2 = torch::empty({8, 4096}, input0.options());
  auto rotated_2 = torch::empty({8, 4096}, input0.options());
  auto attention_2 = torch::empty({8, 2048}, input0.options());
  auto residual_2 = torch::empty({8, 1024}, input0.options());
  auto postnorm_2 = torch::empty({8, 1024}, input0.options());
  auto gate_up_2 = torch::empty({8, 6144}, input0.options());
  auto activated_2 = torch::empty({8, 3072}, input0.options());
  auto hidden_2 = torch::empty({8, 1024}, input0.options());
  auto normalized_3 = torch::empty({8, 1024}, input0.options());
  auto qkv_3 = torch::empty({8, 4096}, input0.options());
  auto rotated_3 = torch::empty({8, 4096}, input0.options());
  auto attention_3 = torch::empty({8, 2048}, input0.options());
  auto residual_3 = torch::empty({8, 1024}, input0.options());
  auto postnorm_3 = torch::empty({8, 1024}, input0.options());
  auto gate_up_3 = torch::empty({8, 6144}, input0.options());
  auto activated_3 = torch::empty({8, 3072}, input0.options());
  auto hidden_3 = torch::empty({8, 1024}, input0.options());
  auto normalized_4 = torch::empty({8, 1024}, input0.options());
  auto qkv_4 = torch::empty({8, 4096}, input0.options());
  auto rotated_4 = torch::empty({8, 4096}, input0.options());
  auto attention_4 = torch::empty({8, 2048}, input0.options());
  auto residual_4 = torch::empty({8, 1024}, input0.options());
  auto postnorm_4 = torch::empty({8, 1024}, input0.options());
  auto gate_up_4 = torch::empty({8, 6144}, input0.options());
  auto activated_4 = torch::empty({8, 3072}, input0.options());
  auto hidden_4 = torch::empty({8, 1024}, input0.options());
  auto normalized_5 = torch::empty({8, 1024}, input0.options());
  auto qkv_5 = torch::empty({8, 4096}, input0.options());
  auto rotated_5 = torch::empty({8, 4096}, input0.options());
  auto attention_5 = torch::empty({8, 2048}, input0.options());
  auto residual_5 = torch::empty({8, 1024}, input0.options());
  auto postnorm_5 = torch::empty({8, 1024}, input0.options());
  auto gate_up_5 = torch::empty({8, 6144}, input0.options());
  auto activated_5 = torch::empty({8, 3072}, input0.options());
  auto hidden_5 = torch::empty({8, 1024}, input0.options());
  auto normalized_6 = torch::empty({8, 1024}, input0.options());
  auto qkv_6 = torch::empty({8, 4096}, input0.options());
  auto rotated_6 = torch::empty({8, 4096}, input0.options());
  auto attention_6 = torch::empty({8, 2048}, input0.options());
  auto residual_6 = torch::empty({8, 1024}, input0.options());
  auto postnorm_6 = torch::empty({8, 1024}, input0.options());
  auto gate_up_6 = torch::empty({8, 6144}, input0.options());
  auto activated_6 = torch::empty({8, 3072}, input0.options());
  auto hidden_6 = torch::empty({8, 1024}, input0.options());
  auto normalized_7 = torch::empty({8, 1024}, input0.options());
  auto qkv_7 = torch::empty({8, 4096}, input0.options());
  auto rotated_7 = torch::empty({8, 4096}, input0.options());
  auto attention_7 = torch::empty({8, 2048}, input0.options());
  auto residual_7 = torch::empty({8, 1024}, input0.options());
  auto postnorm_7 = torch::empty({8, 1024}, input0.options());
  auto gate_up_7 = torch::empty({8, 6144}, input0.options());
  auto activated_7 = torch::empty({8, 3072}, input0.options());
  auto hidden_7 = torch::empty({8, 1024}, input0.options());
  auto normalized_8 = torch::empty({8, 1024}, input0.options());
  auto qkv_8 = torch::empty({8, 4096}, input0.options());
  auto rotated_8 = torch::empty({8, 4096}, input0.options());
  auto attention_8 = torch::empty({8, 2048}, input0.options());
  auto residual_8 = torch::empty({8, 1024}, input0.options());
  auto postnorm_8 = torch::empty({8, 1024}, input0.options());
  auto gate_up_8 = torch::empty({8, 6144}, input0.options());
  auto activated_8 = torch::empty({8, 3072}, input0.options());
  auto hidden_8 = torch::empty({8, 1024}, input0.options());
  auto normalized_9 = torch::empty({8, 1024}, input0.options());
  auto qkv_9 = torch::empty({8, 4096}, input0.options());
  auto rotated_9 = torch::empty({8, 4096}, input0.options());
  auto attention_9 = torch::empty({8, 2048}, input0.options());
  auto residual_9 = torch::empty({8, 1024}, input0.options());
  auto postnorm_9 = torch::empty({8, 1024}, input0.options());
  auto gate_up_9 = torch::empty({8, 6144}, input0.options());
  auto activated_9 = torch::empty({8, 3072}, input0.options());
  auto hidden_9 = torch::empty({8, 1024}, input0.options());
  auto normalized_10 = torch::empty({8, 1024}, input0.options());
  auto qkv_10 = torch::empty({8, 4096}, input0.options());
  auto rotated_10 = torch::empty({8, 4096}, input0.options());
  auto attention_10 = torch::empty({8, 2048}, input0.options());
  auto residual_10 = torch::empty({8, 1024}, input0.options());
  auto postnorm_10 = torch::empty({8, 1024}, input0.options());
  auto gate_up_10 = torch::empty({8, 6144}, input0.options());
  auto activated_10 = torch::empty({8, 3072}, input0.options());
  auto hidden_10 = torch::empty({8, 1024}, input0.options());
  auto normalized_11 = torch::empty({8, 1024}, input0.options());
  auto qkv_11 = torch::empty({8, 4096}, input0.options());
  auto rotated_11 = torch::empty({8, 4096}, input0.options());
  auto attention_11 = torch::empty({8, 2048}, input0.options());
  auto residual_11 = torch::empty({8, 1024}, input0.options());
  auto postnorm_11 = torch::empty({8, 1024}, input0.options());
  auto gate_up_11 = torch::empty({8, 6144}, input0.options());
  auto activated_11 = torch::empty({8, 3072}, input0.options());
  auto hidden_11 = torch::empty({8, 1024}, input0.options());
  auto normalized_12 = torch::empty({8, 1024}, input0.options());
  auto qkv_12 = torch::empty({8, 4096}, input0.options());
  auto rotated_12 = torch::empty({8, 4096}, input0.options());
  auto attention_12 = torch::empty({8, 2048}, input0.options());
  auto residual_12 = torch::empty({8, 1024}, input0.options());
  auto postnorm_12 = torch::empty({8, 1024}, input0.options());
  auto gate_up_12 = torch::empty({8, 6144}, input0.options());
  auto activated_12 = torch::empty({8, 3072}, input0.options());
  auto hidden_12 = torch::empty({8, 1024}, input0.options());
  auto normalized_13 = torch::empty({8, 1024}, input0.options());
  auto qkv_13 = torch::empty({8, 4096}, input0.options());
  auto rotated_13 = torch::empty({8, 4096}, input0.options());
  auto attention_13 = torch::empty({8, 2048}, input0.options());
  auto residual_13 = torch::empty({8, 1024}, input0.options());
  auto postnorm_13 = torch::empty({8, 1024}, input0.options());
  auto gate_up_13 = torch::empty({8, 6144}, input0.options());
  auto activated_13 = torch::empty({8, 3072}, input0.options());
  auto hidden_13 = torch::empty({8, 1024}, input0.options());
  auto normalized_14 = torch::empty({8, 1024}, input0.options());
  auto qkv_14 = torch::empty({8, 4096}, input0.options());
  auto rotated_14 = torch::empty({8, 4096}, input0.options());
  auto attention_14 = torch::empty({8, 2048}, input0.options());
  auto residual_14 = torch::empty({8, 1024}, input0.options());
  auto postnorm_14 = torch::empty({8, 1024}, input0.options());
  auto gate_up_14 = torch::empty({8, 6144}, input0.options());
  auto activated_14 = torch::empty({8, 3072}, input0.options());
  auto hidden_14 = torch::empty({8, 1024}, input0.options());
  auto normalized_15 = torch::empty({8, 1024}, input0.options());
  auto qkv_15 = torch::empty({8, 4096}, input0.options());
  auto rotated_15 = torch::empty({8, 4096}, input0.options());
  auto attention_15 = torch::empty({8, 2048}, input0.options());
  auto residual_15 = torch::empty({8, 1024}, input0.options());
  auto postnorm_15 = torch::empty({8, 1024}, input0.options());
  auto gate_up_15 = torch::empty({8, 6144}, input0.options());
  auto activated_15 = torch::empty({8, 3072}, input0.options());
  auto hidden_15 = torch::empty({8, 1024}, input0.options());
  auto normalized_16 = torch::empty({8, 1024}, input0.options());
  auto qkv_16 = torch::empty({8, 4096}, input0.options());
  auto rotated_16 = torch::empty({8, 4096}, input0.options());
  auto attention_16 = torch::empty({8, 2048}, input0.options());
  auto residual_16 = torch::empty({8, 1024}, input0.options());
  auto postnorm_16 = torch::empty({8, 1024}, input0.options());
  auto gate_up_16 = torch::empty({8, 6144}, input0.options());
  auto activated_16 = torch::empty({8, 3072}, input0.options());
  auto hidden_16 = torch::empty({8, 1024}, input0.options());
  auto normalized_17 = torch::empty({8, 1024}, input0.options());
  auto qkv_17 = torch::empty({8, 4096}, input0.options());
  auto rotated_17 = torch::empty({8, 4096}, input0.options());
  auto attention_17 = torch::empty({8, 2048}, input0.options());
  auto residual_17 = torch::empty({8, 1024}, input0.options());
  auto postnorm_17 = torch::empty({8, 1024}, input0.options());
  auto gate_up_17 = torch::empty({8, 6144}, input0.options());
  auto activated_17 = torch::empty({8, 3072}, input0.options());
  auto hidden_17 = torch::empty({8, 1024}, input0.options());
  auto normalized_18 = torch::empty({8, 1024}, input0.options());
  auto qkv_18 = torch::empty({8, 4096}, input0.options());
  auto rotated_18 = torch::empty({8, 4096}, input0.options());
  auto attention_18 = torch::empty({8, 2048}, input0.options());
  auto residual_18 = torch::empty({8, 1024}, input0.options());
  auto postnorm_18 = torch::empty({8, 1024}, input0.options());
  auto gate_up_18 = torch::empty({8, 6144}, input0.options());
  auto activated_18 = torch::empty({8, 3072}, input0.options());
  auto hidden_18 = torch::empty({8, 1024}, input0.options());
  auto normalized_19 = torch::empty({8, 1024}, input0.options());
  auto qkv_19 = torch::empty({8, 4096}, input0.options());
  auto rotated_19 = torch::empty({8, 4096}, input0.options());
  auto attention_19 = torch::empty({8, 2048}, input0.options());
  auto residual_19 = torch::empty({8, 1024}, input0.options());
  auto postnorm_19 = torch::empty({8, 1024}, input0.options());
  auto gate_up_19 = torch::empty({8, 6144}, input0.options());
  auto activated_19 = torch::empty({8, 3072}, input0.options());
  auto hidden_19 = torch::empty({8, 1024}, input0.options());
  auto normalized_20 = torch::empty({8, 1024}, input0.options());
  auto qkv_20 = torch::empty({8, 4096}, input0.options());
  auto rotated_20 = torch::empty({8, 4096}, input0.options());
  auto attention_20 = torch::empty({8, 2048}, input0.options());
  auto residual_20 = torch::empty({8, 1024}, input0.options());
  auto postnorm_20 = torch::empty({8, 1024}, input0.options());
  auto gate_up_20 = torch::empty({8, 6144}, input0.options());
  auto activated_20 = torch::empty({8, 3072}, input0.options());
  auto hidden_20 = torch::empty({8, 1024}, input0.options());
  auto normalized_21 = torch::empty({8, 1024}, input0.options());
  auto qkv_21 = torch::empty({8, 4096}, input0.options());
  auto rotated_21 = torch::empty({8, 4096}, input0.options());
  auto attention_21 = torch::empty({8, 2048}, input0.options());
  auto residual_21 = torch::empty({8, 1024}, input0.options());
  auto postnorm_21 = torch::empty({8, 1024}, input0.options());
  auto gate_up_21 = torch::empty({8, 6144}, input0.options());
  auto activated_21 = torch::empty({8, 3072}, input0.options());
  auto hidden_21 = torch::empty({8, 1024}, input0.options());
  auto normalized_22 = torch::empty({8, 1024}, input0.options());
  auto qkv_22 = torch::empty({8, 4096}, input0.options());
  auto rotated_22 = torch::empty({8, 4096}, input0.options());
  auto attention_22 = torch::empty({8, 2048}, input0.options());
  auto residual_22 = torch::empty({8, 1024}, input0.options());
  auto postnorm_22 = torch::empty({8, 1024}, input0.options());
  auto gate_up_22 = torch::empty({8, 6144}, input0.options());
  auto activated_22 = torch::empty({8, 3072}, input0.options());
  auto hidden_22 = torch::empty({8, 1024}, input0.options());
  auto normalized_23 = torch::empty({8, 1024}, input0.options());
  auto qkv_23 = torch::empty({8, 4096}, input0.options());
  auto rotated_23 = torch::empty({8, 4096}, input0.options());
  auto attention_23 = torch::empty({8, 2048}, input0.options());
  auto residual_23 = torch::empty({8, 1024}, input0.options());
  auto postnorm_23 = torch::empty({8, 1024}, input0.options());
  auto gate_up_23 = torch::empty({8, 6144}, input0.options());
  auto activated_23 = torch::empty({8, 3072}, input0.options());
  auto hidden_23 = torch::empty({8, 1024}, input0.options());
  auto normalized_24 = torch::empty({8, 1024}, input0.options());
  auto qkv_24 = torch::empty({8, 4096}, input0.options());
  auto rotated_24 = torch::empty({8, 4096}, input0.options());
  auto attention_24 = torch::empty({8, 2048}, input0.options());
  auto residual_24 = torch::empty({8, 1024}, input0.options());
  auto postnorm_24 = torch::empty({8, 1024}, input0.options());
  auto gate_up_24 = torch::empty({8, 6144}, input0.options());
  auto activated_24 = torch::empty({8, 3072}, input0.options());
  auto hidden_24 = torch::empty({8, 1024}, input0.options());
  auto normalized_25 = torch::empty({8, 1024}, input0.options());
  auto qkv_25 = torch::empty({8, 4096}, input0.options());
  auto rotated_25 = torch::empty({8, 4096}, input0.options());
  auto attention_25 = torch::empty({8, 2048}, input0.options());
  auto residual_25 = torch::empty({8, 1024}, input0.options());
  auto postnorm_25 = torch::empty({8, 1024}, input0.options());
  auto gate_up_25 = torch::empty({8, 6144}, input0.options());
  auto activated_25 = torch::empty({8, 3072}, input0.options());
  auto hidden_25 = torch::empty({8, 1024}, input0.options());
  auto normalized_26 = torch::empty({8, 1024}, input0.options());
  auto qkv_26 = torch::empty({8, 4096}, input0.options());
  auto rotated_26 = torch::empty({8, 4096}, input0.options());
  auto attention_26 = torch::empty({8, 2048}, input0.options());
  auto residual_26 = torch::empty({8, 1024}, input0.options());
  auto postnorm_26 = torch::empty({8, 1024}, input0.options());
  auto gate_up_26 = torch::empty({8, 6144}, input0.options());
  auto activated_26 = torch::empty({8, 3072}, input0.options());
  auto hidden_26 = torch::empty({8, 1024}, input0.options());
  auto normalized_27 = torch::empty({8, 1024}, input0.options());
  auto qkv_27 = torch::empty({8, 4096}, input0.options());
  auto rotated_27 = torch::empty({8, 4096}, input0.options());
  auto attention_27 = torch::empty({8, 2048}, input0.options());
  auto residual_27 = torch::empty({8, 1024}, input0.options());
  auto postnorm_27 = torch::empty({8, 1024}, input0.options());
  auto gate_up_27 = torch::empty({8, 6144}, input0.options());
  auto activated_27 = torch::empty({8, 3072}, input0.options());
  auto hidden_27 = torch::empty({8, 1024}, input0.options());
  auto hidden_states = torch::empty({7, 1024}, input0.options());
  auto logits = torch::empty({8, 153600}, input0.options());
  auto tokens = torch::empty({1, 8}, input1.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  embed_rope<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), embedding.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>());
  rms_norm<<<1, 1, 0, stream>>>(embedding.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_0.data_ptr<at::BFloat16>(), 155583488, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), embedding.data_ptr<at::BFloat16>(), qkv_0.data_ptr<at::BFloat16>(), 155585536, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_0.data_ptr<at::BFloat16>(), 161876992);
  attention_mma<<<1, 32, 0, stream>>>(rotated_0.data_ptr<at::BFloat16>(), attention_0.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), embedding.data_ptr<at::BFloat16>(), residual_0.data_ptr<at::BFloat16>(), 159779840, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_0.data_ptr<at::BFloat16>(), 155584512, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_0.data_ptr<at::BFloat16>(), gate_up_0.data_ptr<at::BFloat16>(), 161877248, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_0.data_ptr<at::BFloat16>(), activated_0.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_0.data_ptr<at::BFloat16>(), hidden_0.data_ptr<at::BFloat16>(), 168168704, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_1.data_ptr<at::BFloat16>(), 171314432, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_0.data_ptr<at::BFloat16>(), qkv_1.data_ptr<at::BFloat16>(), 171316480, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_1.data_ptr<at::BFloat16>(), 177607936);
  attention_mma<<<1, 32, 0, stream>>>(rotated_1.data_ptr<at::BFloat16>(), attention_1.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_0.data_ptr<at::BFloat16>(), residual_1.data_ptr<at::BFloat16>(), 175510784, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_1.data_ptr<at::BFloat16>(), 171315456, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_1.data_ptr<at::BFloat16>(), gate_up_1.data_ptr<at::BFloat16>(), 177608192, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_1.data_ptr<at::BFloat16>(), activated_1.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_1.data_ptr<at::BFloat16>(), hidden_1.data_ptr<at::BFloat16>(), 183899648, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_2.data_ptr<at::BFloat16>(), 187045376, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_1.data_ptr<at::BFloat16>(), qkv_2.data_ptr<at::BFloat16>(), 187047424, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_2.data_ptr<at::BFloat16>(), 193338880);
  attention_mma<<<1, 32, 0, stream>>>(rotated_2.data_ptr<at::BFloat16>(), attention_2.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_1.data_ptr<at::BFloat16>(), residual_2.data_ptr<at::BFloat16>(), 191241728, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_2.data_ptr<at::BFloat16>(), 187046400, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_2.data_ptr<at::BFloat16>(), gate_up_2.data_ptr<at::BFloat16>(), 193339136, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_2.data_ptr<at::BFloat16>(), activated_2.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_2.data_ptr<at::BFloat16>(), hidden_2.data_ptr<at::BFloat16>(), 199630592, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_3.data_ptr<at::BFloat16>(), 202776320, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_2.data_ptr<at::BFloat16>(), qkv_3.data_ptr<at::BFloat16>(), 202778368, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_3.data_ptr<at::BFloat16>(), 209069824);
  attention_mma<<<1, 32, 0, stream>>>(rotated_3.data_ptr<at::BFloat16>(), attention_3.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_2.data_ptr<at::BFloat16>(), residual_3.data_ptr<at::BFloat16>(), 206972672, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_3.data_ptr<at::BFloat16>(), 202777344, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_3.data_ptr<at::BFloat16>(), gate_up_3.data_ptr<at::BFloat16>(), 209070080, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_3.data_ptr<at::BFloat16>(), activated_3.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_3.data_ptr<at::BFloat16>(), hidden_3.data_ptr<at::BFloat16>(), 215361536, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_4.data_ptr<at::BFloat16>(), 218507264, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_3.data_ptr<at::BFloat16>(), qkv_4.data_ptr<at::BFloat16>(), 218509312, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_4.data_ptr<at::BFloat16>(), 224800768);
  attention_mma<<<1, 32, 0, stream>>>(rotated_4.data_ptr<at::BFloat16>(), attention_4.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_3.data_ptr<at::BFloat16>(), residual_4.data_ptr<at::BFloat16>(), 222703616, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_4.data_ptr<at::BFloat16>(), 218508288, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_4.data_ptr<at::BFloat16>(), gate_up_4.data_ptr<at::BFloat16>(), 224801024, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_4.data_ptr<at::BFloat16>(), activated_4.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_4.data_ptr<at::BFloat16>(), hidden_4.data_ptr<at::BFloat16>(), 231092480, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_5.data_ptr<at::BFloat16>(), 234238208, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_4.data_ptr<at::BFloat16>(), qkv_5.data_ptr<at::BFloat16>(), 234240256, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_5.data_ptr<at::BFloat16>(), 240531712);
  attention_mma<<<1, 32, 0, stream>>>(rotated_5.data_ptr<at::BFloat16>(), attention_5.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_4.data_ptr<at::BFloat16>(), residual_5.data_ptr<at::BFloat16>(), 238434560, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_5.data_ptr<at::BFloat16>(), 234239232, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_5.data_ptr<at::BFloat16>(), gate_up_5.data_ptr<at::BFloat16>(), 240531968, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_5.data_ptr<at::BFloat16>(), activated_5.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_5.data_ptr<at::BFloat16>(), hidden_5.data_ptr<at::BFloat16>(), 246823424, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_6.data_ptr<at::BFloat16>(), 249969152, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_5.data_ptr<at::BFloat16>(), qkv_6.data_ptr<at::BFloat16>(), 249971200, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_6.data_ptr<at::BFloat16>(), 256262656);
  attention_mma<<<1, 32, 0, stream>>>(rotated_6.data_ptr<at::BFloat16>(), attention_6.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_5.data_ptr<at::BFloat16>(), residual_6.data_ptr<at::BFloat16>(), 254165504, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_6.data_ptr<at::BFloat16>(), 249970176, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_6.data_ptr<at::BFloat16>(), gate_up_6.data_ptr<at::BFloat16>(), 256262912, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_6.data_ptr<at::BFloat16>(), activated_6.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_6.data_ptr<at::BFloat16>(), hidden_6.data_ptr<at::BFloat16>(), 262554368, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_7.data_ptr<at::BFloat16>(), 265700096, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_6.data_ptr<at::BFloat16>(), qkv_7.data_ptr<at::BFloat16>(), 265702144, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_7.data_ptr<at::BFloat16>(), 271993600);
  attention_mma<<<1, 32, 0, stream>>>(rotated_7.data_ptr<at::BFloat16>(), attention_7.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_6.data_ptr<at::BFloat16>(), residual_7.data_ptr<at::BFloat16>(), 269896448, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_7.data_ptr<at::BFloat16>(), 265701120, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_7.data_ptr<at::BFloat16>(), gate_up_7.data_ptr<at::BFloat16>(), 271993856, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_7.data_ptr<at::BFloat16>(), activated_7.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_7.data_ptr<at::BFloat16>(), hidden_7.data_ptr<at::BFloat16>(), 278285312, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_8.data_ptr<at::BFloat16>(), 281431040, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_7.data_ptr<at::BFloat16>(), qkv_8.data_ptr<at::BFloat16>(), 281433088, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_8.data_ptr<at::BFloat16>(), 287724544);
  attention_mma<<<1, 32, 0, stream>>>(rotated_8.data_ptr<at::BFloat16>(), attention_8.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_7.data_ptr<at::BFloat16>(), residual_8.data_ptr<at::BFloat16>(), 285627392, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_8.data_ptr<at::BFloat16>(), 281432064, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_8.data_ptr<at::BFloat16>(), gate_up_8.data_ptr<at::BFloat16>(), 287724800, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_8.data_ptr<at::BFloat16>(), activated_8.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_8.data_ptr<at::BFloat16>(), hidden_8.data_ptr<at::BFloat16>(), 294016256, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_9.data_ptr<at::BFloat16>(), 297161984, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_8.data_ptr<at::BFloat16>(), qkv_9.data_ptr<at::BFloat16>(), 297164032, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_9.data_ptr<at::BFloat16>(), 303455488);
  attention_mma<<<1, 32, 0, stream>>>(rotated_9.data_ptr<at::BFloat16>(), attention_9.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_8.data_ptr<at::BFloat16>(), residual_9.data_ptr<at::BFloat16>(), 301358336, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_9.data_ptr<at::BFloat16>(), 297163008, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_9.data_ptr<at::BFloat16>(), gate_up_9.data_ptr<at::BFloat16>(), 303455744, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_9.data_ptr<at::BFloat16>(), activated_9.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_9.data_ptr<at::BFloat16>(), hidden_9.data_ptr<at::BFloat16>(), 309747200, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_10.data_ptr<at::BFloat16>(), 312892928, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_9.data_ptr<at::BFloat16>(), qkv_10.data_ptr<at::BFloat16>(), 312894976, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_10.data_ptr<at::BFloat16>(), 319186432);
  attention_mma<<<1, 32, 0, stream>>>(rotated_10.data_ptr<at::BFloat16>(), attention_10.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_9.data_ptr<at::BFloat16>(), residual_10.data_ptr<at::BFloat16>(), 317089280, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_10.data_ptr<at::BFloat16>(), 312893952, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_10.data_ptr<at::BFloat16>(), gate_up_10.data_ptr<at::BFloat16>(), 319186688, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_10.data_ptr<at::BFloat16>(), activated_10.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_10.data_ptr<at::BFloat16>(), hidden_10.data_ptr<at::BFloat16>(), 325478144, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_11.data_ptr<at::BFloat16>(), 328623872, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_10.data_ptr<at::BFloat16>(), qkv_11.data_ptr<at::BFloat16>(), 328625920, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_11.data_ptr<at::BFloat16>(), 334917376);
  attention_mma<<<1, 32, 0, stream>>>(rotated_11.data_ptr<at::BFloat16>(), attention_11.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_10.data_ptr<at::BFloat16>(), residual_11.data_ptr<at::BFloat16>(), 332820224, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_11.data_ptr<at::BFloat16>(), 328624896, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_11.data_ptr<at::BFloat16>(), gate_up_11.data_ptr<at::BFloat16>(), 334917632, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_11.data_ptr<at::BFloat16>(), activated_11.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_11.data_ptr<at::BFloat16>(), hidden_11.data_ptr<at::BFloat16>(), 341209088, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_12.data_ptr<at::BFloat16>(), 344354816, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_11.data_ptr<at::BFloat16>(), qkv_12.data_ptr<at::BFloat16>(), 344356864, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_12.data_ptr<at::BFloat16>(), 350648320);
  attention_mma<<<1, 32, 0, stream>>>(rotated_12.data_ptr<at::BFloat16>(), attention_12.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_11.data_ptr<at::BFloat16>(), residual_12.data_ptr<at::BFloat16>(), 348551168, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_12.data_ptr<at::BFloat16>(), 344355840, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_12.data_ptr<at::BFloat16>(), gate_up_12.data_ptr<at::BFloat16>(), 350648576, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_12.data_ptr<at::BFloat16>(), activated_12.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_12.data_ptr<at::BFloat16>(), hidden_12.data_ptr<at::BFloat16>(), 356940032, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_13.data_ptr<at::BFloat16>(), 360085760, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_12.data_ptr<at::BFloat16>(), qkv_13.data_ptr<at::BFloat16>(), 360087808, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_13.data_ptr<at::BFloat16>(), 366379264);
  attention_mma<<<1, 32, 0, stream>>>(rotated_13.data_ptr<at::BFloat16>(), attention_13.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_12.data_ptr<at::BFloat16>(), residual_13.data_ptr<at::BFloat16>(), 364282112, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_13.data_ptr<at::BFloat16>(), 360086784, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_13.data_ptr<at::BFloat16>(), gate_up_13.data_ptr<at::BFloat16>(), 366379520, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_13.data_ptr<at::BFloat16>(), activated_13.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_13.data_ptr<at::BFloat16>(), hidden_13.data_ptr<at::BFloat16>(), 372670976, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_14.data_ptr<at::BFloat16>(), 375816704, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_13.data_ptr<at::BFloat16>(), qkv_14.data_ptr<at::BFloat16>(), 375818752, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_14.data_ptr<at::BFloat16>(), 382110208);
  attention_mma<<<1, 32, 0, stream>>>(rotated_14.data_ptr<at::BFloat16>(), attention_14.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_13.data_ptr<at::BFloat16>(), residual_14.data_ptr<at::BFloat16>(), 380013056, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_14.data_ptr<at::BFloat16>(), 375817728, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_14.data_ptr<at::BFloat16>(), gate_up_14.data_ptr<at::BFloat16>(), 382110464, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_14.data_ptr<at::BFloat16>(), activated_14.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_14.data_ptr<at::BFloat16>(), hidden_14.data_ptr<at::BFloat16>(), 388401920, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_15.data_ptr<at::BFloat16>(), 391547648, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_14.data_ptr<at::BFloat16>(), qkv_15.data_ptr<at::BFloat16>(), 391549696, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_15.data_ptr<at::BFloat16>(), 397841152);
  attention_mma<<<1, 32, 0, stream>>>(rotated_15.data_ptr<at::BFloat16>(), attention_15.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_14.data_ptr<at::BFloat16>(), residual_15.data_ptr<at::BFloat16>(), 395744000, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_15.data_ptr<at::BFloat16>(), 391548672, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_15.data_ptr<at::BFloat16>(), gate_up_15.data_ptr<at::BFloat16>(), 397841408, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_15.data_ptr<at::BFloat16>(), activated_15.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_15.data_ptr<at::BFloat16>(), hidden_15.data_ptr<at::BFloat16>(), 404132864, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_16.data_ptr<at::BFloat16>(), 407278592, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_15.data_ptr<at::BFloat16>(), qkv_16.data_ptr<at::BFloat16>(), 407280640, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_16.data_ptr<at::BFloat16>(), 413572096);
  attention_mma<<<1, 32, 0, stream>>>(rotated_16.data_ptr<at::BFloat16>(), attention_16.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_15.data_ptr<at::BFloat16>(), residual_16.data_ptr<at::BFloat16>(), 411474944, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_16.data_ptr<at::BFloat16>(), 407279616, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_16.data_ptr<at::BFloat16>(), gate_up_16.data_ptr<at::BFloat16>(), 413572352, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_16.data_ptr<at::BFloat16>(), activated_16.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_16.data_ptr<at::BFloat16>(), hidden_16.data_ptr<at::BFloat16>(), 419863808, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_17.data_ptr<at::BFloat16>(), 423009536, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_16.data_ptr<at::BFloat16>(), qkv_17.data_ptr<at::BFloat16>(), 423011584, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_17.data_ptr<at::BFloat16>(), 429303040);
  attention_mma<<<1, 32, 0, stream>>>(rotated_17.data_ptr<at::BFloat16>(), attention_17.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_16.data_ptr<at::BFloat16>(), residual_17.data_ptr<at::BFloat16>(), 427205888, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_17.data_ptr<at::BFloat16>(), 423010560, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_17.data_ptr<at::BFloat16>(), gate_up_17.data_ptr<at::BFloat16>(), 429303296, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_17.data_ptr<at::BFloat16>(), activated_17.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_17.data_ptr<at::BFloat16>(), hidden_17.data_ptr<at::BFloat16>(), 435594752, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_18.data_ptr<at::BFloat16>(), 438740480, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_17.data_ptr<at::BFloat16>(), qkv_18.data_ptr<at::BFloat16>(), 438742528, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_18.data_ptr<at::BFloat16>(), 445033984);
  attention_mma<<<1, 32, 0, stream>>>(rotated_18.data_ptr<at::BFloat16>(), attention_18.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_17.data_ptr<at::BFloat16>(), residual_18.data_ptr<at::BFloat16>(), 442936832, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_18.data_ptr<at::BFloat16>(), 438741504, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_18.data_ptr<at::BFloat16>(), gate_up_18.data_ptr<at::BFloat16>(), 445034240, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_18.data_ptr<at::BFloat16>(), activated_18.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_18.data_ptr<at::BFloat16>(), hidden_18.data_ptr<at::BFloat16>(), 451325696, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_19.data_ptr<at::BFloat16>(), 454471424, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_18.data_ptr<at::BFloat16>(), qkv_19.data_ptr<at::BFloat16>(), 454473472, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_19.data_ptr<at::BFloat16>(), 460764928);
  attention_mma<<<1, 32, 0, stream>>>(rotated_19.data_ptr<at::BFloat16>(), attention_19.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_18.data_ptr<at::BFloat16>(), residual_19.data_ptr<at::BFloat16>(), 458667776, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_19.data_ptr<at::BFloat16>(), 454472448, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_19.data_ptr<at::BFloat16>(), gate_up_19.data_ptr<at::BFloat16>(), 460765184, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_19.data_ptr<at::BFloat16>(), activated_19.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_19.data_ptr<at::BFloat16>(), hidden_19.data_ptr<at::BFloat16>(), 467056640, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_20.data_ptr<at::BFloat16>(), 470202368, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_19.data_ptr<at::BFloat16>(), qkv_20.data_ptr<at::BFloat16>(), 470204416, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_20.data_ptr<at::BFloat16>(), 476495872);
  attention_mma<<<1, 32, 0, stream>>>(rotated_20.data_ptr<at::BFloat16>(), attention_20.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_19.data_ptr<at::BFloat16>(), residual_20.data_ptr<at::BFloat16>(), 474398720, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_20.data_ptr<at::BFloat16>(), 470203392, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_20.data_ptr<at::BFloat16>(), gate_up_20.data_ptr<at::BFloat16>(), 476496128, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_20.data_ptr<at::BFloat16>(), activated_20.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_20.data_ptr<at::BFloat16>(), hidden_20.data_ptr<at::BFloat16>(), 482787584, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_21.data_ptr<at::BFloat16>(), 485933312, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_20.data_ptr<at::BFloat16>(), qkv_21.data_ptr<at::BFloat16>(), 485935360, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_21.data_ptr<at::BFloat16>(), 492226816);
  attention_mma<<<1, 32, 0, stream>>>(rotated_21.data_ptr<at::BFloat16>(), attention_21.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_20.data_ptr<at::BFloat16>(), residual_21.data_ptr<at::BFloat16>(), 490129664, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_21.data_ptr<at::BFloat16>(), 485934336, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_21.data_ptr<at::BFloat16>(), gate_up_21.data_ptr<at::BFloat16>(), 492227072, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_21.data_ptr<at::BFloat16>(), activated_21.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_21.data_ptr<at::BFloat16>(), hidden_21.data_ptr<at::BFloat16>(), 498518528, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_22.data_ptr<at::BFloat16>(), 501664256, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_21.data_ptr<at::BFloat16>(), qkv_22.data_ptr<at::BFloat16>(), 501666304, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_22.data_ptr<at::BFloat16>(), 507957760);
  attention_mma<<<1, 32, 0, stream>>>(rotated_22.data_ptr<at::BFloat16>(), attention_22.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_21.data_ptr<at::BFloat16>(), residual_22.data_ptr<at::BFloat16>(), 505860608, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_22.data_ptr<at::BFloat16>(), 501665280, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_22.data_ptr<at::BFloat16>(), gate_up_22.data_ptr<at::BFloat16>(), 507958016, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_22.data_ptr<at::BFloat16>(), activated_22.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_22.data_ptr<at::BFloat16>(), hidden_22.data_ptr<at::BFloat16>(), 514249472, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_23.data_ptr<at::BFloat16>(), 517395200, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_22.data_ptr<at::BFloat16>(), qkv_23.data_ptr<at::BFloat16>(), 517397248, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_23.data_ptr<at::BFloat16>(), 523688704);
  attention_mma<<<1, 32, 0, stream>>>(rotated_23.data_ptr<at::BFloat16>(), attention_23.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_22.data_ptr<at::BFloat16>(), residual_23.data_ptr<at::BFloat16>(), 521591552, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_23.data_ptr<at::BFloat16>(), 517396224, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_23.data_ptr<at::BFloat16>(), gate_up_23.data_ptr<at::BFloat16>(), 523688960, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_23.data_ptr<at::BFloat16>(), activated_23.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_23.data_ptr<at::BFloat16>(), hidden_23.data_ptr<at::BFloat16>(), 529980416, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_24.data_ptr<at::BFloat16>(), 533126144, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_23.data_ptr<at::BFloat16>(), qkv_24.data_ptr<at::BFloat16>(), 533128192, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_24.data_ptr<at::BFloat16>(), 539419648);
  attention_mma<<<1, 32, 0, stream>>>(rotated_24.data_ptr<at::BFloat16>(), attention_24.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_23.data_ptr<at::BFloat16>(), residual_24.data_ptr<at::BFloat16>(), 537322496, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_24.data_ptr<at::BFloat16>(), 533127168, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_24.data_ptr<at::BFloat16>(), gate_up_24.data_ptr<at::BFloat16>(), 539419904, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_24.data_ptr<at::BFloat16>(), activated_24.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_24.data_ptr<at::BFloat16>(), hidden_24.data_ptr<at::BFloat16>(), 545711360, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_25.data_ptr<at::BFloat16>(), 548857088, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_24.data_ptr<at::BFloat16>(), qkv_25.data_ptr<at::BFloat16>(), 548859136, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_25.data_ptr<at::BFloat16>(), 555150592);
  attention_mma<<<1, 32, 0, stream>>>(rotated_25.data_ptr<at::BFloat16>(), attention_25.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_24.data_ptr<at::BFloat16>(), residual_25.data_ptr<at::BFloat16>(), 553053440, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_25.data_ptr<at::BFloat16>(), 548858112, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_25.data_ptr<at::BFloat16>(), gate_up_25.data_ptr<at::BFloat16>(), 555150848, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_25.data_ptr<at::BFloat16>(), activated_25.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_25.data_ptr<at::BFloat16>(), hidden_25.data_ptr<at::BFloat16>(), 561442304, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_26.data_ptr<at::BFloat16>(), 564588032, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_25.data_ptr<at::BFloat16>(), qkv_26.data_ptr<at::BFloat16>(), 564590080, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_26.data_ptr<at::BFloat16>(), 570881536);
  attention_mma<<<1, 32, 0, stream>>>(rotated_26.data_ptr<at::BFloat16>(), attention_26.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_25.data_ptr<at::BFloat16>(), residual_26.data_ptr<at::BFloat16>(), 568784384, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_26.data_ptr<at::BFloat16>(), 564589056, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_26.data_ptr<at::BFloat16>(), gate_up_26.data_ptr<at::BFloat16>(), 570881792, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_26.data_ptr<at::BFloat16>(), activated_26.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_26.data_ptr<at::BFloat16>(), hidden_26.data_ptr<at::BFloat16>(), 577173248, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), normalized_27.data_ptr<at::BFloat16>(), 580318976, 8);
  linear_mma<<<1, 32, 0, stream>>>(normalized_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_26.data_ptr<at::BFloat16>(), qkv_27.data_ptr<at::BFloat16>(), 580321024, 4096, 1024, 8, 1, 0);
  norm_rope<<<1, 1, 0, stream>>>(qkv_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(), rotated_27.data_ptr<at::BFloat16>(), 586612480);
  attention_mma<<<1, 32, 0, stream>>>(rotated_27.data_ptr<at::BFloat16>(), attention_27.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(attention_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_26.data_ptr<at::BFloat16>(), residual_27.data_ptr<at::BFloat16>(), 584515328, 1024, 2048, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(residual_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), postnorm_27.data_ptr<at::BFloat16>(), 580320000, 8);
  linear_mma<<<1, 32, 0, stream>>>(postnorm_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_27.data_ptr<at::BFloat16>(), gate_up_27.data_ptr<at::BFloat16>(), 586612736, 6144, 1024, 8, 2, 0);
  silu_mul<<<1, 1, 0, stream>>>(gate_up_27.data_ptr<at::BFloat16>(), activated_27.data_ptr<at::BFloat16>());
  linear_mma<<<1, 32, 0, stream>>>(activated_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual_27.data_ptr<at::BFloat16>(), hidden_27.data_ptr<at::BFloat16>(), 592904192, 1024, 3072, 8, 0, 1);
  rms_norm<<<1, 1, 0, stream>>>(hidden_27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_states.data_ptr<at::BFloat16>(), 155582464, 7);
  linear_mma<<<1, 32, 0, stream>>>(hidden_states.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden_27.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), 0, 153600, 1024, 7, 3, 0);
  select_token<<<1, 1, 0, stream>>>(logits.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), tokens.data_ptr<int64_t>());
  return {tokens, hidden_states};
}
