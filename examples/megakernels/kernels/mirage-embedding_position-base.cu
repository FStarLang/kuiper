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

__global__ void embedding(const at::BFloat16* weights, const int64_t* tokens,
                          at::BFloat16* hidden, at::BFloat16* rotary, int position) {
    for (int row = 0; row < 7; ++row)
        for (int col = 0; col < 1024; ++col)
            hidden[row * 1024 + col] = weights[tokens[row] * 1024 + col];
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 128; ++col) {
            float power = powf(1000000.0f, float(col % 64) / 64.0f);
            float angle = __fmul_rn(float(row + position), 1.0f / power);
            rotary[row * 128 + col] = at::BFloat16(cosf(angle));
            rotary[896 + row * 128 + col] = at::BFloat16(sinf(angle));
        }
    }
}


}  // namespace

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    int64_t input2) {
    TORCH_CHECK(input0.is_cuda() && input0.is_contiguous() &&
                input0.scalar_type() == at::kBFloat16 &&
                input0.dim() == 2 && input0.size(0) == 151936 && input0.size(1) == 1024,
                "Expected a contiguous CUDA BF16 [151936, 1024] embedding table");
    TORCH_CHECK(input1.is_cuda() && input1.is_contiguous() &&
                input1.scalar_type() == at::kLong && input1.device() == input0.device() &&
                input1.dim() == 1 && input1.size(0) == 7,
                "Expected seven contiguous CUDA int64 token IDs on the weights device");
    TORCH_CHECK(input2 >= 0 && input2 <= std::numeric_limits<int>::max() - 6,
                "The seven positions must fit in a nonnegative CUDA int");
    const c10::cuda::CUDAGuard guard(input0.device());
    auto hidden = torch::empty({7, 1024}, input0.options());
    auto rotary = torch::empty({2, 7, 128}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    embedding<<<1, 1, 0, stream>>>(
        input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        hidden.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(),
        static_cast<int>(input2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {hidden, rotary};
}
