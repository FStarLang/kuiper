#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

__global__ void router_gemm(
    const at::BFloat16* tokens,
    const at::BFloat16* weights,
    float* scores,
    int token_count,
    int hidden_count) {
    const int lane = static_cast<int>(threadIdx.x);
    const int group = lane / 4;
    const int pair = (lane % 4) * 2;

    for (int token_base = 0; token_base < token_count; token_base += 16) {
        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;
        for (int hidden_tile = 0; hidden_tile < hidden_count; hidden_tile += 64) {
            for (int hidden_step = 0; hidden_step < 64; hidden_step += 16) {
                const int hidden = hidden_tile + hidden_step + pair;
                const int a_index = (token_base + group) * hidden_count + hidden;
                const int b_index = group * hidden_count + hidden;

                const uint32_t a0 = static_cast<uint32_t>(tokens[a_index].x)
                    | (static_cast<uint32_t>(tokens[a_index + 1].x) << 16);
                const uint32_t a1 = static_cast<uint32_t>(tokens[a_index + 8 * hidden_count].x)
                    | (static_cast<uint32_t>(tokens[a_index + 8 * hidden_count + 1].x) << 16);
                const uint32_t a2 = static_cast<uint32_t>(tokens[a_index + 8].x)
                    | (static_cast<uint32_t>(tokens[a_index + 9].x) << 16);
                const uint32_t a3 = static_cast<uint32_t>(tokens[a_index + 8 * hidden_count + 8].x)
                    | (static_cast<uint32_t>(tokens[a_index + 8 * hidden_count + 9].x) << 16);
                const uint32_t b0 = static_cast<uint32_t>(weights[b_index].x)
                    | (static_cast<uint32_t>(weights[b_index + 1].x) << 16);
                const uint32_t b1 = static_cast<uint32_t>(weights[b_index + 8].x)
                    | (static_cast<uint32_t>(weights[b_index + 9].x) << 16);

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

__global__ void router_softmax(
    const float* scores,
    float* normalized_rows,
    int token_count) {
    const float negative_infinity = __uint_as_float(0xff800000u);
    for (int token = 0; token < token_count; ++token) {
        float row[8];
        for (int expert = 0; expert < 8; ++expert) {
            row[expert] = scores[token * 8 + expert];
        }

        float maximum = negative_infinity;
        float denominator = 0.0f;
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
        for (int expert = 0; expert < 8; ++expert) {
            normalized_rows[token * 8 + expert] = row[expert];
        }
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1) {
    TORCH_CHECK(input0.is_cuda() && input1.is_cuda(), "Tokens and weights must be CUDA tensors.");
    TORCH_CHECK(input0.device() == input1.device(), "Tokens and weights must be on the same device.");
    TORCH_CHECK(input0.scalar_type() == at::kBFloat16 && input1.scalar_type() == at::kBFloat16,
                "Tokens and weights must be BF16.");
    TORCH_CHECK(input0.is_contiguous() && input1.is_contiguous(), "Tokens and weights must be contiguous.");
    TORCH_CHECK(input0.dim() == 2 && input1.dim() == 2, "Tokens and weights must be matrices.");
    TORCH_CHECK(input0.size(0) > 0 && input0.size(0) % 16 == 0,
                "The token count must be a positive multiple of 16.");
    TORCH_CHECK(input0.size(1) > 0 && input0.size(1) % 64 == 0,
                "The hidden dimension must be a positive multiple of 64.");
    TORCH_CHECK(input1.size(0) == 8 && input1.size(1) == input0.size(1),
                "Weights must have expert-major shape [8, hidden].");
    TORCH_CHECK(input0.numel() <= std::numeric_limits<int>::max(), "Token indexing must fit int32.");

    const c10::cuda::CUDAGuard device_guard(input0.device());
    const int token_count = static_cast<int>(input0.size(0));
    const int hidden_count = static_cast<int>(input0.size(1));
    auto scores = torch::empty({token_count, 8}, input0.options().dtype(at::kFloat));
    auto normalized_rows = torch::empty({token_count, 8}, input0.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    router_gemm<<<1, 32, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
        scores.data_ptr<float>(), token_count, hidden_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    router_softmax<<<1, 1, 0, stream>>>(scores.data_ptr<float>(), normalized_rows.data_ptr<float>(), token_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {normalized_rows};
}
