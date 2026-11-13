#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <cmath>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <initializer_list>

namespace {

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

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                  std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
}

}  // namespace

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                      torch::Tensor input2, int64_t input3) {
  check_tensor(input0, at::kBFloat16, {8, 1024}, "input0");
  check_tensor(input1, at::kBFloat16, {15730944}, "input1");
  check_tensor(input2, at::kBFloat16, {2, 4096, 128}, "input2");
  TORCH_CHECK(input0.device() == input1.device() && input0.device() == input2.device(),
              "inputs must share a CUDA device");
  TORCH_CHECK(input3 >= 0 && input3 <= 4089, "seven sequence positions must fit the RoPE table");
  const c10::cuda::CUDAGuard guard(input0.device());
  auto normalized = torch::empty({8, 1024}, input0.options());
  auto qkv = torch::empty({8, 4096}, input0.options());
  auto rotated = torch::empty({8, 4096}, input0.options());
  auto attention = torch::empty({8, 2048}, input0.options());
  auto residual = torch::empty({8, 1024}, input0.options());
  auto postnorm = torch::empty({8, 1024}, input0.options());
  auto gate_up = torch::empty({8, 6144}, input0.options());
  auto activated = torch::empty({8, 3072}, input0.options());
  auto hidden = torch::empty({8, 1024}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  rms_norm<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), 0, 8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_mma<<<1, 32, 0, stream>>>(normalized.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
      qkv.data_ptr<at::BFloat16>(), 2048, 4096, 1024, 8, 1, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  norm_rope<<<1, 1, 0, stream>>>(qkv.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), input2.data_ptr<at::BFloat16>() + input3 * 128,
      rotated.data_ptr<at::BFloat16>(), 6293504);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  attention_mma<<<1, 32, 0, stream>>>(rotated.data_ptr<at::BFloat16>(),
      attention.data_ptr<at::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_mma<<<1, 32, 0, stream>>>(attention.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
      residual.data_ptr<at::BFloat16>(), 4196352, 1024, 2048, 8, 0, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  rms_norm<<<1, 1, 0, stream>>>(residual.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), postnorm.data_ptr<at::BFloat16>(), 1024, 8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_mma<<<1, 32, 0, stream>>>(postnorm.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), residual.data_ptr<at::BFloat16>(),
      gate_up.data_ptr<at::BFloat16>(), 6293760, 6144, 1024, 8, 2, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  silu_mul<<<1, 1, 0, stream>>>(gate_up.data_ptr<at::BFloat16>(),
      activated.data_ptr<at::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_mma<<<1, 32, 0, stream>>>(activated.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), residual.data_ptr<at::BFloat16>(),
      hidden.data_ptr<at::BFloat16>(), 12585216, 1024, 3072, 8, 0, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {hidden};
}
