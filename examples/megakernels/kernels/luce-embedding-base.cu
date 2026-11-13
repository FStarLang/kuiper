#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
__device__ void position_embedding(at::BFloat16* h, int position) {
    float inverse[32] = {
        0x1.000000p+0f, 0x1.356656p-1f, 0x1.75f036p-2f, 0x1.c3f06ap-3f,
        0x1.111aeep-3f, 0x1.4a12acp-4f, 0x1.8eec7ep-5f, 0x1.e222eep-6f,
        0x1.235a72p-6f, 0x1.6020a4p-7f, 0x1.a9942ap-8f, 0x1.012cfap-8f,
        0x1.36d21ap-9f, 0x1.77a7dap-10f, 0x1.c603c4p-11f, 0x1.125c04p-11f,
        0x1.4b96c0p-12f, 0x1.90c182p-13f, 0x1.e459c4p-14f, 0x1.24b0fcp-14f,
        0x1.61bea4p-15f, 0x1.ab8882p-16f, 0x1.025b56p-16f, 0x1.383f88p-17f,
        0x1.796182p-18f, 0x1.c8198cp-19f, 0x1.139e94p-19f, 0x1.4d1c98p-20f,
        0x1.9298aep-21f, 0x1.e6933ap-22f, 0x1.26091cp-22f, 0x1.635e88p-23f
    };
    for (int i = 0; i < 32; ++i) {
        float angle = __fmul_rn(inverse[i], float(position));
        h[15600 + i] = at::BFloat16(cosf(angle));
        h[15632 + i] = at::BFloat16(sinf(angle));
    }
}

void check_tensor(const torch::Tensor& value, const torch::Tensor& owner,
                  at::ScalarType dtype, int64_t count, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.numel() == count, name, ": invalid device, layout, dtype or size");
    TORCH_CHECK(value.device() == owner.device(), name, ": tensors must share a CUDA device");
}

__global__ void embedding_kernel(const at::BFloat16* weight, const int64_t* prompt,
                                const int64_t* previous, at::BFloat16* output,
                                at::BFloat16* rope, float* initial_state,
                                at::BFloat16* initial_conv, int prompt_length, int position) {
    at::BFloat16 h[16000];
    if (position == 0) {
        for (int i = 0; i < 4718592; ++i) initial_state[i] = 0.0f;
        for (int i = 0; i < 442368; ++i) initial_conv[i] = at::BFloat16(0.0f);
    }
    int64_t next_token = position == 0 ? 0 : previous[0];
    int64_t token = position < prompt_length ? prompt[position] : next_token;
    for (int i = 0; i < 1024; ++i) h[i] = weight[token * 1024 + i];
    position_embedding(h, position);
    for (int i = 0; i < 1024; ++i) output[i] = h[i];
    for (int i = 0; i < 64; ++i) rope[i] = h[15600 + i];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, int64_t input3) {
    check_tensor(input0, input0, at::kBFloat16, 254279680, "embedding weights");
    check_tensor(input1, input0, at::kLong, input1.numel(), "prompt");
    check_tensor(input2, input0, at::kLong, 1, "previous next_token");
    TORCH_CHECK(input1.numel() > 0 && input1.numel() <= 2048 && input3 >= 0 && input3 < 2048,
                "invalid prompt length or position");
    c10::cuda::CUDAGuard guard(input0.device());
    auto output = torch::empty({1024}, input0.options());
    auto rope = torch::empty({64}, input0.options());
    auto state = torch::empty({input3 == 0 ? 4718592 : 0}, input0.options().dtype(at::kFloat));
    auto conv = torch::empty({input3 == 0 ? 442368 : 0}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    embedding_kernel<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        input2.data_ptr<int64_t>(), output.data_ptr<at::BFloat16>(), rope.data_ptr<at::BFloat16>(),
        state.data_ptr<float>(), conv.data_ptr<at::BFloat16>(), input1.numel(), input3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, rope, state, conv};
}
