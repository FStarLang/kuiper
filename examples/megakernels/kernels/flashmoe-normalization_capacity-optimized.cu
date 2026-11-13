#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

__global__ void router_selected_weights(
    const float* normalized_rows,
    const int32_t* selected_experts,
    const float* best_values,
    int32_t* route_selected,
    float* route_probabilities,
    int token_count) {
    for (int token = 0; token < token_count; ++token) {
        float row[8];
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = normalized_rows[token * 8 + expert];
        }
        const int selected_expert = selected_experts[token];
        const float best = best_values[token];
        const float selected_sum = __fadd_rn(0.0f, best);
        for (int expert = 0; expert < 8; ++expert) {
            const bool selected = expert == selected_expert;
            route_selected[token * 8 + expert] = selected ? 1 : 0;
            route_probabilities[token * 8 + expert] = selected
                ? fdividef(row[expert], selected_sum) : 0.0f;
        }
    }
}

__global__ void capacity_observe(
    const int32_t* route_selected,
    const float* route_probabilities,
    int32_t* expert_demand,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival,
    int token_count,
    int capacity) {
    for (int expert = 0; expert < 8; ++expert) {
        for (int slot = 0; slot < capacity; ++slot) {
            token_indices[expert * capacity + slot] = -1;
            probabilities[expert * capacity + slot] = 0.0f;
        }

        int32_t count = 0;
        for (int token = 0; token < token_count; ++token) {
            const int index = token * 8 + expert;
            const int32_t selected = route_selected[index];
            count += selected;
            const int32_t slot = count - 1;
            const bool keep = selected != 0 && slot < capacity;
            survival[index] = keep ? 1 : 0;
            if (keep) {
                token_indices[expert * capacity + slot] = static_cast<int64_t>(token);
                probabilities[expert * capacity + slot] = route_probabilities[index];
            }
        }
        expert_demand[expert] = count;
    }
}

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    int64_t input3) {
    TORCH_CHECK(input0.is_cuda() && input1.is_cuda() && input2.is_cuda(),
                "Rows, selected experts and best values must be CUDA tensors.");
    TORCH_CHECK(input0.device() == input1.device() && input0.device() == input2.device(),
                "All routing inputs must be on the same device.");
    TORCH_CHECK(input0.scalar_type() == at::kFloat && input1.scalar_type() == at::kInt
                    && input2.scalar_type() == at::kFloat,
                "Routing dtypes must be float32, int32, float32.");
    TORCH_CHECK(input0.is_contiguous() && input1.is_contiguous() && input2.is_contiguous(),
                "All routing inputs must be contiguous.");
    TORCH_CHECK(input0.dim() == 2 && input0.size(1) == 8,
                "Rows must have shape [tokens, 8].");
    TORCH_CHECK(input0.size(0) > 0 && input0.numel() <= std::numeric_limits<int>::max(),
                "The token count must be positive and row indexing must fit int32.");
    TORCH_CHECK(input1.dim() == 1 && input2.dim() == 1
                    && input1.size(0) == input0.size(0) && input2.size(0) == input0.size(0),
                "Selected experts and best values must have shape [tokens].");
    TORCH_CHECK(input3 > 0 && input3 <= std::numeric_limits<int>::max() / 8,
                "Capacity must be positive and expert-slot indexing must fit int32.");

    const c10::cuda::CUDAGuard device_guard(input0.device());
    const int token_count = static_cast<int>(input0.size(0));
    const int capacity = static_cast<int>(input3);
    auto route_selected = torch::empty({token_count, 8}, input0.options().dtype(at::kInt));
    auto route_probabilities = torch::empty({token_count, 8}, input0.options());
    auto expert_demand = torch::empty({8}, input0.options().dtype(at::kInt));
    auto token_indices = torch::empty({8, capacity}, input0.options().dtype(at::kLong));
    auto probabilities = torch::empty({8, capacity}, input0.options());
    auto survival = torch::empty({token_count, 8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    router_selected_weights<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), input1.data_ptr<int32_t>(), input2.data_ptr<float>(),
        route_selected.data_ptr<int32_t>(), route_probabilities.data_ptr<float>(), token_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    capacity_observe<<<1, 1, 0, stream>>>(
        route_selected.data_ptr<int32_t>(), route_probabilities.data_ptr<float>(),
        expert_demand.data_ptr<int32_t>(), token_indices.data_ptr<int64_t>(),
        probabilities.data_ptr<float>(), survival.data_ptr<int32_t>(), token_count, capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {expert_demand, token_indices, probabilities, survival};
}
