#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>

namespace {

__global__ void select_token(const at::BFloat16* logits, const int64_t* tokens,
                            int64_t* output, int score_row) {
    for (int i = 0; i < 7; ++i) output[i] = tokens[i];
    float best = float(logits[score_row * 151936]);
    int index = 0;
    for (int i = 1; i < 151936; ++i) {
        float value = float(logits[score_row * 151936 + i]);
        if (!isnan(best) && (isnan(value) || value > best)) {
            best = value;
            index = i;
        }
    }
    output[7] = index;
}


}  // namespace

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    int64_t input2) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                input0.scalar_type() == at::kBFloat16 &&
                input0.dim() == 2 && input0.size(0) == 7 && input0.size(1) == 151936,
                "Expected the complete contiguous CUDA BF16 [7, 151936] logits");
    TORCH_CHECK(input1.is_cuda() && input1.is_contiguous() &&
                input1.scalar_type() == at::kLong && input1.device() == input0.device() &&
                input1.dim() == 1 && input1.size(0) == 7,
                "Expected seven contiguous CUDA int64 prefix tokens on the logits device");
    TORCH_CHECK(input2 >= 0 && input2 < 7, "The selected logits row must be in [0, 7)");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto output = torch::empty({1, 8}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    select_token<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(), static_cast<int>(input2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output};
}
