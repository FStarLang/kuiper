#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                  std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
}

}  // namespace

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                      torch::Tensor input2) {
  check_tensor(input0, at::kBFloat16, {8, 1024}, "input0");
  check_tensor(input1, at::kBFloat16, {1024}, "input1");
  check_tensor(input2, at::kBFloat16, {151936, 1024}, "input2");
  TORCH_CHECK(input0.device() == input1.device() && input0.device() == input2.device(),
              "inputs must share a CUDA device");
  const c10::cuda::CUDAGuard guard(input0.device());
  auto hidden_states = torch::empty({7, 1024}, input0.options());
  auto logits = torch::empty({8, 153600}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  rms_norm<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
      input1.data_ptr<at::BFloat16>(), hidden_states.data_ptr<at::BFloat16>(), 0, 7);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_mma<<<1, 32, 0, stream>>>(hidden_states.data_ptr<at::BFloat16>(),
      input2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
      logits.data_ptr<at::BFloat16>(), 0, 153600, 1024, 7, 3, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The accepted allocation's eighth row is neither written nor read.
  return {hidden_states, logits.narrow(0, 0, 7)};
}
