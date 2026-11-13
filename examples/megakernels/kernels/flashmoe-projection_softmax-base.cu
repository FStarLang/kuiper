#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void check_tensor(
    const torch::Tensor& value,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const c10::Device& device,
    const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == device,
                name, " must be on the token's CUDA device");
    TORCH_CHECK(value.scalar_type() == dtype && value.is_contiguous() && value.sizes() == shape,
                name, " has an invalid dtype, shape or layout");
}

__global__ void projection_softmax(
    const at::BFloat16* token_values,
    const at::BFloat16* weights,
    const int64_t* previous_counts,
    const int64_t* previous_indices,
    const float* previous_probabilities,
    const int32_t* previous_survival,
    int token,
    float* output_row,
    int64_t* prefix_counts,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival) {
    if (token == 0) {
        for (int expert = 0; expert < 8; ++expert) {
            prefix_counts[expert] = 0;
        }
        for (int i = 0; i < 256; ++i) {
            token_indices[i] = -1;
            probabilities[i] = 0.0f;
        }
    } else {
        for (int expert = 0; expert < 8; ++expert) {
            prefix_counts[expert] = previous_counts[expert];
        }
        for (int i = 0; i < 256; ++i) {
            token_indices[i] = previous_indices[i];
            probabilities[i] = previous_probabilities[i];
        }
        for (int i = 0; i < token * 8; ++i) {
            survival[i] = previous_survival[i];
        }
    }

    float row[8];
    for (int expert = 0; expert < 8; ++expert) {
        float score = 0.0f;
        for (int hidden = 0; hidden < 512; ++hidden) {
            const float x = static_cast<float>(token_values[hidden]);
            const float w = static_cast<float>(weights[expert * 512 + hidden]);
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
    for (int expert = 0; expert < 8; ++expert) {
        output_row[expert] = row[expert];
    }
}

} // namespace

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor input3,
    torch::Tensor input4,
    torch::Tensor input5,
    int64_t input6) {
    TORCH_CHECK(input6 >= 0 && input6 < 256, "token position must be in [0, 256)");
    check_tensor(input0, at::kBFloat16, {512}, input0.device(), "input0");
    check_tensor(input1, at::kBFloat16, {8, 512}, input0.device(), "input1");
    if (input6 == 0) {
        for (const auto& state : {input2, input3, input4, input5}) {
            TORCH_CHECK(state.is_cuda() && state.device() == input0.device()
                        && state.is_contiguous() && state.numel() == 0,
                        "initial-state arguments must be empty CUDA tensors on the token's device");
        }
    } else {
        check_tensor(input2, at::kLong, {8}, input0.device(), "input2");
        check_tensor(input3, at::kLong, {8, 32}, input0.device(), "input3");
        check_tensor(input4, at::kFloat, {8, 32}, input0.device(), "input4");
        check_tensor(input5, at::kInt, {input6, 8}, input0.device(), "input5");
    }
    const c10::cuda::CUDAGuard guard(input0.device());
    auto row = torch::empty({8}, input0.options().dtype(at::kFloat));
    auto prefix_counts = torch::empty({8}, input0.options().dtype(at::kLong));
    auto token_indices = torch::empty({8, 32}, input0.options().dtype(at::kLong));
    auto probabilities = torch::empty({8, 32}, input0.options().dtype(at::kFloat));
    auto survival = torch::empty({input6 + 1, 8}, input0.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    projection_softmax<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<at::BFloat16>(),
        input6 == 0 ? nullptr : input2.data_ptr<int64_t>(),
        input6 == 0 ? nullptr : input3.data_ptr<int64_t>(),
        input6 == 0 ? nullptr : input4.data_ptr<float>(),
        input6 == 0 ? nullptr : input5.data_ptr<int32_t>(),
        static_cast<int>(input6), row.data_ptr<float>(), prefix_counts.data_ptr<int64_t>(),
        token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(),
        survival.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {row, prefix_counts, token_indices, probabilities, survival};
}
