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

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                  std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
}

}  // namespace

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
  check_tensor(input0, at::kBFloat16, {151936, 1024}, "input0");
  check_tensor(input1, at::kLong, {7}, "input1");
  TORCH_CHECK(input0.device() == input1.device(), "inputs must share a CUDA device");
  const c10::cuda::CUDAGuard guard(input0.device());
  auto embedding = torch::empty({8, 1024}, input0.options());
  auto rope = torch::empty({2, 4096, 128}, input0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  embed_rope<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
      input1.data_ptr<int64_t>(), embedding.data_ptr<at::BFloat16>(),
      rope.data_ptr<at::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {embedding, rope};
}
