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

__global__ void select_token(const at::BFloat16* logits, const int64_t* prompt,
                             int64_t* output, int64_t logit_row) {
    float value[256], chunk_value[96];
    int64_t index[256], chunk_index[96];
    float initial = -CUDART_INF_F;
    for (int chunk = 0; chunk < 96; ++chunk) {
      for (int thread = 0; thread < 128; ++thread) {
        value[thread] = initial;
        index[thread] = -1;
        for (int i = thread; i < 1600 && chunk * 1600 + i < 151936; i += 128) {
          float v = float(logits[logit_row * 153600 + chunk * 1600 + i]);
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

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                  std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
}

}  // namespace

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                      int64_t input2) {
  check_tensor(input0, at::kBFloat16, {7, 153600}, "input0");
  check_tensor(input1, at::kLong, {7}, "input1");
  TORCH_CHECK(input0.device() == input1.device(), "inputs must share a CUDA device");
  TORCH_CHECK(input2 >= 0 && input2 < 7, "selected logit row must be in [0, 7)");
  const c10::cuda::CUDAGuard guard(input0.device());
  auto tokens = torch::empty({1, 8}, input1.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  select_token<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
      input1.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(), input2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {tokens};
}
