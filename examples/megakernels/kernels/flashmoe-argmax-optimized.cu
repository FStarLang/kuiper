#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <vector>

__global__ void router_argmax(
    const float* normalized_rows,
    int32_t* selected_experts,
    float* best_values,
    int token_count) {
    const float negative_infinity = __uint_as_float(0xff800000u);
    for (int token = 0; token < token_count; ++token) {
        float row[8];
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = normalized_rows[token * 8 + expert];
        }
        float best = negative_infinity;
        int selected_expert = 0;
        for (int expert = 0; expert < 8; ++expert) {
            if (row[expert] > best) {
                best = row[expert];
                selected_expert = expert;
            }
        }
        selected_experts[token] = selected_expert;
        best_values[token] = best;
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0) {
    TORCH_CHECK(input0.is_cuda() && input0.scalar_type() == at::kFloat,
                "Argmax rows must be CUDA float32.");
    TORCH_CHECK(input0.is_contiguous() && input0.dim() == 2 && input0.size(1) == 8,
                "Argmax rows must be contiguous with shape [tokens, 8].");
    TORCH_CHECK(input0.size(0) > 0 && input0.numel() <= std::numeric_limits<int>::max(),
                "The token count must be positive and row indexing must fit int32.");

    const c10::cuda::CUDAGuard device_guard(input0.device());
    const int token_count = static_cast<int>(input0.size(0));
    auto selected_experts = torch::empty({token_count}, input0.options().dtype(at::kInt));
    auto best_values = torch::empty({token_count}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    router_argmax<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), selected_experts.data_ptr<int32_t>(),
        best_values.data_ptr<float>(), token_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {selected_experts, best_values};
}
