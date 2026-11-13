#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void check_tensor(const torch::Tensor& value, const torch::Tensor& owner,
                  at::ScalarType dtype, int64_t count, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.numel() == count, name, ": invalid device, layout, dtype or size");
    TORCH_CHECK(value.device() == owner.device(), name, ": tensors must share a CUDA device");
}

__global__ void argmax_kernel(const at::BFloat16* scores, const int64_t* previous,
                              int64_t* selected, int64_t* tokens, int position, int prompt_length) {
    float best_logit = -INFINITY;
    int64_t next_token = 0;
    for (int row = 0; row < 248320; row += 16)
        for (int i = 0; i < 16; ++i) {
            float value = float(scores[row + i]);
            if (value > best_logit || (isnan(value) && !isnan(best_logit))) {
                best_logit = value;
                next_token = row + i;
            }
        }
    selected[0] = next_token;
    int previous_count = position >= prompt_length ? position - prompt_length + 1 : 0;
    for (int i = 0; i < previous_count; ++i) tokens[i] = previous[i];
    if (position >= prompt_length - 1)
        tokens[position - prompt_length + 1] = next_token;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    int64_t input2, int64_t input3) {
    TORCH_CHECK(input2 >= 0 && input2 < 2048 && input3 > 0 && input3 <= 2048,
                "invalid position or prompt length");
    int64_t previous_count = input2 >= input3 ? input2 - input3 + 1 : 0;
    int64_t count = input2 >= input3 - 1 ? input2 - input3 + 2 : 0;
    check_tensor(input0, input0, at::kBFloat16, 248320, "rounded projection scores");
    check_tensor(input1, input0, at::kLong, previous_count, "emitted token prefix");
    c10::cuda::CUDAGuard guard(input0.device());
    auto selected = torch::empty({}, input1.options());
    auto tokens = torch::empty({count}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    argmax_kernel<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        selected.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(), input2, input3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {selected, tokens};
}
