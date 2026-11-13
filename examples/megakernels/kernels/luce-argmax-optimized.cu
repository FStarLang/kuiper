#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <cmath>

__device__ bool argmax_better(float value, int index, float best, int best_index) {
    return index >= 0 && (best_index < 0 || value > best ||
                         (isnan(value) && !isnan(best)));
}

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

__global__ void ordered_argmax(const float* scores, const int64_t* previous_history,
                               int64_t position, int64_t prompt_length,
                               int64_t* output_token, int64_t* token_history) {
    float block_values[512];
    int block_indices[512];
    int token;
    for (int block = 0; block < 512; ++block) {
        float values[32];
        int indices[32];
        for (int warp = 0; warp < 32; ++warp) {
            float maximum = -INFINITY;
            int index = -1;
            if (warp < 8)
                for (int row = block * 485 + warp; row < (block + 1) * 485; row += 8) {
                    float score = scores[row];
                    if (argmax_better(score, row, maximum, index)) { maximum = score; index = row; }
                }
            values[warp] = maximum;
            indices[warp] = index;
        }
        for (int distance = 16; distance > 0; distance /= 2)
            for (int lane = 0; lane < distance; ++lane)
                if (argmax_better(values[lane + distance], indices[lane + distance],
                                  values[lane], indices[lane])) {
                    values[lane] = values[lane + distance];
                    indices[lane] = indices[lane + distance];
                }
        block_values[block] = values[0];
        block_indices[block] = indices[0];
    }
    for (int distance = 256; distance > 0; distance /= 2)
        for (int lane = 0; lane < distance; ++lane)
            if (argmax_better(block_values[lane + distance], block_indices[lane + distance],
                              block_values[lane], block_indices[lane])) {
                block_values[lane] = block_values[lane + distance];
                block_indices[lane] = block_indices[lane + distance];
            }
    token = block_indices[0];

    output_token[0] = token;
    int64_t previous_count = position < prompt_length ? 0 : position - prompt_length + 1;
    for (int64_t i = 0; i < previous_count; ++i) token_history[i] = previous_history[i];
    if (position >= prompt_length - 1) token_history[previous_count] = token;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    int64_t input2, int64_t input3) {
    check_tensor(input0, at::kFloat, 248320, "projection scores");
    check_position(input2, input3);
    int64_t previous_count = input2 < input3 ? 0 : input2 - input3 + 1;
    int64_t count = input2 < input3 - 1 ? 0 : input2 - input3 + 2;
    check_tensor(input1, at::kLong, previous_count, "previous token history");
    TORCH_CHECK(input1.dim() == 1, "Invalid token history rank");
    auto token = torch::empty({}, input1.options());
    auto history = torch::empty({count}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    ordered_argmax<<<1, 1, 0, stream>>>(input0.data_ptr<float>(), input1.data_ptr<int64_t>(),
        input2, input3, token.data_ptr<int64_t>(), history.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {token, history};
}
