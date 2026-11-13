#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

__global__ void route_capacity(
    const at::BFloat16* input0,
    const at::BFloat16* input1,
    int32_t* expert_demand,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival) {
    int64_t prefix_counts[8] = {};

    for (int i = 0; i < 256; ++i) {
        token_indices[i] = -1;
        probabilities[i] = 0.0f;
    }

    for (int token = 0; token < 256; ++token) {
        float row[8];
        for (int expert = 0; expert < 8; ++expert) {
            float score = 0.0f;
            for (int hidden = 0; hidden < 512; ++hidden) {
                const float x = static_cast<float>(input0[token * 512 + hidden]);
                const float w = static_cast<float>(input1[expert * 512 + hidden]);
                score = __fmaf_rn(x, w, score);
            }
            row[expert] = score;
            survival[token * 8 + expert] = 0;
        }

        float maximum = row[0];
        for (int expert = 1; expert < 8; ++expert) {
            if (row[expert] > maximum) {
                maximum = row[expert];
            }
        }

        float exponential_sum = 0.0f;
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = expf(__fsub_rn(row[expert], maximum));
            exponential_sum = __fadd_rn(exponential_sum, row[expert]);
        }
        const float inverse_sum = __fdiv_rn(1.0f, exponential_sum);
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = __fmul_rn(row[expert], inverse_sum);
        }

        int selected = 0;
        for (int expert = 1; expert < 8; ++expert) {
            // Strict comparison preserves the lowest expert ID on a tie.
            if (row[expert] > row[selected]) {
                selected = expert;
            }
        }
        const float selected_probability = row[selected];
        const float selected_sum = __fadd_rn(0.0f, selected_probability);
        const float combine_probability = __fdiv_rn(selected_probability, selected_sum);

        // Integer cumsum promotes the source's int32 assignments to int64.
        const int64_t rank = prefix_counts[selected] + 1;
        prefix_counts[selected] = rank;
        if (rank <= 32) {
            const int64_t destination = selected * 32 + rank - 1;
            token_indices[destination] = token;
            probabilities[destination] = combine_probability;
            survival[token * 8 + selected] = 1;
        }
    }

    for (int expert = 0; expert < 8; ++expert) {
        expert_demand[expert] = static_cast<int32_t>(prefix_counts[expert]);
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
    auto expert_demand = torch::empty({8}, input0.options().dtype(at::kInt));
    auto token_indices = torch::empty({8, 32}, input0.options().dtype(at::kLong));
    auto probabilities = torch::empty({8, 32}, input0.options().dtype(at::kFloat));
    auto survival = torch::empty({256, 8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    route_capacity<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(), expert_demand.data_ptr<int32_t>(), token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(), survival.data_ptr<int32_t>());
    return {expert_demand, token_indices, probabilities, survival};
}
