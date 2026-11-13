#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <cmath>

#include <c10/cuda/CUDAException.h>

void check_tensor(const torch::Tensor& value, at::ScalarType dtype, int64_t count,
                  const char* name) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() &&
                value.scalar_type() == dtype && value.numel() == count,
                name, ": expected a contiguous CUDA tensor with the declared dtype and size");
}

void check_empty(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.numel() == 0,
                name, ": position zero requires an explicit empty initial-state tensor");
}

void check_position(int64_t position, int64_t prompt_length) {
    TORCH_CHECK(prompt_length > 0 && prompt_length <= 2048 &&
                position >= 0 && position < 2048, "Invalid sequence position or prompt length");
}

__global__ void embedding(const at::BFloat16* weights, const int64_t* prompt,
                          const int64_t* previous_token, int64_t prompt_length,
                          int64_t position, at::BFloat16* hidden) {
    int token = int(previous_token[0]);
    if (position < prompt_length) token = int(prompt[position]);
    for (int i = 0; i < 1024; ++i) hidden[i] = weights[int64_t(token) * 1024 + i];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, int64_t input3) {
    check_tensor(input0, at::kBFloat16, 254279680, "embedding weights");
    check_tensor(input1, at::kLong, input1.numel(), "prompt");
    check_tensor(input2, at::kLong, 1, "previous token");
    TORCH_CHECK(input1.dim() == 1 && input2.dim() == 0, "Invalid prompt or token rank");
    check_position(input3, input1.numel());
    auto hidden = torch::empty({1024}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    embedding<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        input2.data_ptr<int64_t>(), input1.numel(), input3, hidden.data_ptr<at::BFloat16>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {hidden};
}
