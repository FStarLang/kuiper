#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

__global__ void router_gemm(
    const at::BFloat16* input0,
    const at::BFloat16* input1,
    float* scores) {
    const int lane = static_cast<int>(threadIdx.x);
    const int group = lane / 4;
    const int pair = (lane % 4) * 2;

    for (int token_base = 0; token_base < 256; token_base += 16) {
        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;
        for (int hidden_tile = 0; hidden_tile < 512; hidden_tile += 64) {
            for (int hidden_step = 0; hidden_step < 64; hidden_step += 16) {
                const int hidden = hidden_tile + hidden_step + pair;
                const int a_index = (token_base + group) * 512 + hidden;
                const int b_index = group * 512 + hidden;

                // Each register packs two BF16 elements in the MMA operand layout.
                const uint32_t a0 = static_cast<uint32_t>(input0[a_index].x)
                    | (static_cast<uint32_t>(input0[a_index + 1].x) << 16);
                const uint32_t a1 = static_cast<uint32_t>(input0[a_index + 8 * 512].x)
                    | (static_cast<uint32_t>(input0[a_index + 8 * 512 + 1].x) << 16);
                const uint32_t a2 = static_cast<uint32_t>(input0[a_index + 8].x)
                    | (static_cast<uint32_t>(input0[a_index + 9].x) << 16);
                const uint32_t a3 = static_cast<uint32_t>(input0[a_index + 8 * 512 + 8].x)
                    | (static_cast<uint32_t>(input0[a_index + 8 * 512 + 9].x) << 16);
                const uint32_t b0 = static_cast<uint32_t>(input1[b_index].x)
                    | (static_cast<uint32_t>(input1[b_index + 1].x) << 16);
                const uint32_t b1 = static_cast<uint32_t>(input1[b_index + 8].x)
                    | (static_cast<uint32_t>(input1[b_index + 9].x) << 16);

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
            }
        }
        const int destination = (token_base + group) * 8 + pair;
        scores[destination] = c0;
        scores[destination + 1] = c1;
        scores[destination + 8 * 8] = c2;
        scores[destination + 8 * 8 + 1] = c3;
    }
}

__global__ void router_softmax_top1(
    const float* scores,
    int32_t* route_selected,
    float* route_probabilities) {
    const float negative_infinity = __uint_as_float(0xff800000u);
    for (int token = 0; token < 256; ++token) {
        float row[8];
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = scores[token * 8 + expert];
        }

        float maximum = negative_infinity;
        float denominator = 0.0f;
        // Preserve the native online softmax recurrence, not a two-pass sum.
        for (int expert = 0; expert < 8; ++expert) {
            const float previous_maximum = maximum;
            maximum = fmaxf(maximum, row[expert]);
            denominator = __fmaf_rn(
                denominator,
                __expf(__fsub_rn(previous_maximum, maximum)),
                __expf(__fsub_rn(row[expert], maximum)));
        }
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = __fdividef(
                __expf(__fsub_rn(row[expert], maximum)), denominator);
        }

        float best = negative_infinity;
        int selected_expert = 0;
        for (int expert = 0; expert < 8; ++expert) {
            if (row[expert] > best) {
                best = row[expert];
                selected_expert = expert;
            }
        }
        const float selected_sum = __fadd_rn(0.0f, best);
        for (int expert = 0; expert < 8; ++expert) {
            const bool selected = expert == selected_expert;
            // Native flags are 0 or 1, with identical int32 and uint32 bytes.
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
    int32_t* survival) {
    for (int expert = 0; expert < 8; ++expert) {
        for (int slot = 0; slot < 32; ++slot) {
            token_indices[expert * 32 + slot] = -1;
            probabilities[expert * 32 + slot] = 0.0f;
        }

        int32_t count = 0;
        for (int token = 0; token < 256; ++token) {
            const int index = token * 8 + expert;
            const int32_t selected = route_selected[index];
            count += selected;
            const int32_t slot = count - 1;
            const bool keep = selected != 0 && slot < 32;
            survival[index] = keep ? 1 : 0;
            if (keep) {
                token_indices[expert * 32 + slot] = static_cast<int64_t>(token);
                probabilities[expert * 32 + slot] = route_probabilities[index];
            }
        }
        expert_demand[expert] = count;
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
    auto scores = torch::empty({256, 8}, input0.options().dtype(at::kFloat));
    auto route_selected = torch::empty({256, 8}, input0.options().dtype(at::kInt));
    auto route_probabilities = torch::empty({256, 8}, input0.options().dtype(at::kFloat));
    auto expert_demand = torch::empty({8}, input0.options().dtype(at::kInt));
    auto token_indices = torch::empty({8, 32}, input0.options().dtype(at::kLong));
    auto probabilities = torch::empty({8, 32}, input0.options().dtype(at::kFloat));
    auto survival = torch::empty({256, 8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    router_gemm<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(), scores.data_ptr<float>());
    router_softmax_top1<<<1, 1, 0, stream>>>(scores.data_ptr<float>(), route_selected.data_ptr<int32_t>(), route_probabilities.data_ptr<float>());
    capacity_observe<<<1, 1, 0, stream>>>(route_selected.data_ptr<int32_t>(), route_probabilities.data_ptr<float>(), expert_demand.data_ptr<int32_t>(), token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(), survival.data_ptr<int32_t>());
    return {expert_demand, token_indices, probabilities, survival};
}
