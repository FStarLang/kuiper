#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
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
                name, " must be on the row's CUDA device");
    TORCH_CHECK(value.scalar_type() == dtype && value.is_contiguous() && value.sizes() == shape,
                name, " has an invalid dtype, shape or layout");
}

__global__ void select_expert(
    const float* input_row,
    const int64_t* input_counts,
    const int64_t* input_indices,
    const float* input_probabilities,
    const int32_t* input_survival,
    int token,
    int32_t* output_selected,
    float* output_row,
    int64_t* output_counts,
    int64_t* output_indices,
    float* output_probabilities,
    int32_t* output_survival) {
    float row[8];
    for (int expert = 0; expert < 8; ++expert) {
        row[expert] = input_row[expert];
    }
    int selected = 0;
    for (int expert = 1; expert < 8; ++expert) {
        // Strict comparison preserves the lowest expert ID on a tie.
        if (row[expert] > row[selected]) {
            selected = expert;
        }
    }
    output_selected[0] = selected;
    for (int expert = 0; expert < 8; ++expert) {
        output_row[expert] = row[expert];
        output_counts[expert] = input_counts[expert];
    }
    for (int i = 0; i < 256; ++i) {
        output_indices[i] = input_indices[i];
        output_probabilities[i] = input_probabilities[i];
    }
    for (int i = 0; i < (token + 1) * 8; ++i) {
        output_survival[i] = input_survival[i];
    }
}

} // namespace

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor input3,
    torch::Tensor input4,
    int64_t input5) {
    TORCH_CHECK(input5 >= 0 && input5 < 256, "token position must be in [0, 256)");
    check_tensor(input0, at::kFloat, {8}, input0.device(), "input0");
    check_tensor(input1, at::kLong, {8}, input0.device(), "input1");
    check_tensor(input2, at::kLong, {8, 32}, input0.device(), "input2");
    check_tensor(input3, at::kFloat, {8, 32}, input0.device(), "input3");
    check_tensor(input4, at::kInt, {input5 + 1, 8}, input0.device(), "input4");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto selected = torch::empty({}, input0.options().dtype(at::kInt));
    auto row = torch::empty({8}, input0.options());
    auto prefix_counts = torch::empty({8}, input1.options());
    auto token_indices = torch::empty({8, 32}, input2.options());
    auto probabilities = torch::empty({8, 32}, input3.options());
    auto survival = torch::empty({input5 + 1, 8}, input4.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    select_expert<<<1, 1, 0, stream>>>(
        input0.data_ptr<float>(), input1.data_ptr<int64_t>(), input2.data_ptr<int64_t>(),
        input3.data_ptr<float>(), input4.data_ptr<int32_t>(), static_cast<int>(input5),
        selected.data_ptr<int32_t>(), row.data_ptr<float>(), prefix_counts.data_ptr<int64_t>(),
        token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(),
        survival.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {selected, row, prefix_counts, token_indices, probabilities, survival};
}
