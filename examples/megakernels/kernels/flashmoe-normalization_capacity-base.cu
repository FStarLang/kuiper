#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cassert>
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
                name, " must be on the selection's CUDA device");
    TORCH_CHECK(value.scalar_type() == dtype && value.is_contiguous() && value.sizes() == shape,
                name, " has an invalid dtype, shape or layout");
}

__global__ void normalization_capacity_outputs(
    const int32_t* input_selected,
    const float* row,
    const int64_t* input_counts,
    const int64_t* input_indices,
    const float* input_probabilities,
    const int32_t* input_survival,
    int token,
    int64_t* prefix_counts,
    int64_t* token_indices,
    float* probabilities,
    int32_t* survival,
    int32_t* expert_demand) {
    for (int expert = 0; expert < 8; ++expert) {
        prefix_counts[expert] = input_counts[expert];
    }
    for (int i = 0; i < 256; ++i) {
        token_indices[i] = input_indices[i];
        probabilities[i] = input_probabilities[i];
    }
    for (int i = 0; i < (token + 1) * 8; ++i) {
        survival[i] = input_survival[i];
    }

    const int selected = input_selected[0];
    assert(selected >= 0 && selected < 8);
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

    if (token == 255) {
        for (int expert = 0; expert < 8; ++expert) {
            expert_demand[expert] = static_cast<int32_t>(prefix_counts[expert]);
        }
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
    check_tensor(input0, at::kInt, {}, input0.device(), "input0");
    check_tensor(input1, at::kFloat, {8}, input0.device(), "input1");
    check_tensor(input2, at::kLong, {8}, input0.device(), "input2");
    check_tensor(input3, at::kLong, {8, 32}, input0.device(), "input3");
    check_tensor(input4, at::kFloat, {8, 32}, input0.device(), "input4");
    check_tensor(input5, at::kInt, {input6 + 1, 8}, input0.device(), "input5");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto prefix_counts = torch::empty({8}, input2.options());
    auto token_indices = torch::empty({8, 32}, input3.options());
    auto probabilities = torch::empty({8, 32}, input4.options());
    auto survival = torch::empty({input6 + 1, 8}, input5.options());
    auto expert_demand = torch::empty({input6 == 255 ? 8 : 0}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    normalization_capacity_outputs<<<1, 1, 0, stream>>>(
        input0.data_ptr<int32_t>(), input1.data_ptr<float>(), input2.data_ptr<int64_t>(),
        input3.data_ptr<int64_t>(), input4.data_ptr<float>(), input5.data_ptr<int32_t>(),
        static_cast<int>(input6), prefix_counts.data_ptr<int64_t>(),
        token_indices.data_ptr<int64_t>(), probabilities.data_ptr<float>(),
        survival.data_ptr<int32_t>(), expert_demand.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {prefix_counts, token_indices, probabilities, survival, expert_demand};
}
