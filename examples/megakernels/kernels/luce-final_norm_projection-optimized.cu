#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <cmath>

__device__ float add32(float a, float b) {
    float r;
    asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float mul32(float a, float b) {
    float r;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float fma32(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ float reciprocal32(float a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ float rsqrt32(float a) {
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ float exp32(float a) {
    float r;
    float x = mul32(a, 1.44269504088896340736f);
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ float sigmoid32(float a) {
    return reciprocal32(add32(1.0f, exp32(-a)));
}

__device__ float silu32(float a) {
    return mul32(a, sigmoid32(a));
}

__device__ float sum32(float* partial) {
    // Replay the source's shuffle-down tree without physical warp operations.
#pragma unroll
    for (int distance = 16; distance > 0; distance /= 2)
#pragma unroll
        for (int lane = 0; lane < distance; ++lane)
            partial[lane] = add32(partial[lane], partial[lane + distance]);
    return partial[0];
}

__device__ __noinline__ float dot_bf16(const at::BFloat16* weights, const at::BFloat16* x,
                                     int64_t offset) {
    float partial[32];
#pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
        float sum = 0.0f;
        for (int k = lane * 8; k < 1024; k += 256) {
            uint4 w = reinterpret_cast<const uint4*>(weights + offset + k)[0];
            uint4 a = reinterpret_cast<const uint4*>(x + k)[0];
            const at::BFloat16* wp = reinterpret_cast<const at::BFloat16*>(&w);
            const at::BFloat16* ap = reinterpret_cast<const at::BFloat16*>(&a);
            float chunk = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                chunk = fma32(float(wp[j]), float(ap[j]), chunk);
            sum = add32(sum, chunk);
        }
        partial[lane] = sum;
    }
    return sum32(partial);
}

__device__ __noinline__ float dot_float(const at::BFloat16* weights, const float* x,
                                      int64_t offset, int count) {
    float partial[32];
#pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
        float sum = 0.0f;
        for (int k = lane * 8; k < count; k += 256) {
            uint4 w = reinterpret_cast<const uint4*>(weights + offset + k)[0];
            float4 a[2] = {reinterpret_cast<const float4*>(x + k)[0],
                           reinterpret_cast<const float4*>(x + k)[1]};
            const at::BFloat16* wp = reinterpret_cast<const at::BFloat16*>(&w);
            const float* ap = reinterpret_cast<const float*>(a);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum = fma32(float(wp[j]), ap[j], sum);
        }
        partial[lane] = sum;
    }
    return sum32(partial);
}

__device__ float hidden_rstd(const at::BFloat16* x) {
    float warps[32], lanes[32];
    for (int warp = 0; warp < 16; ++warp) {
        for (int lane = 0; lane < 32; ++lane) {
            float a = float(x[warp * 32 + lane]);
            float b = float(x[warp * 32 + lane + 512]);
            lanes[lane] = fma32(b, b, fma32(a, a, 0.0f));
        }
        warps[warp] = sum32(lanes);
    }
    for (int warp = 16; warp < 32; ++warp) warps[warp] = 0.0f;
    return rsqrt32(fma32(sum32(warps), 1.0f / 1024.0f, 1e-6f));
}

__device__ float strided_dot(const float* a, const float* b, int count) {
    float partial[32];
    for (int lane = 0; lane < 32; ++lane) {
        float sum = 0.0f;
        for (int i = lane; i < count; i += 32)
            sum = fma32(a[i], b[i], sum);
        partial[lane] = sum;
    }
    return sum32(partial);
}

__device__ float gated_rstd(const float* x) {
    float warps[32], lanes[32];
    for (int warp = 0; warp < 4; ++warp) {
        for (int lane = 0; lane < 32; ++lane) {
            float v = x[warp * 32 + lane];
            lanes[lane] = fma32(v, v, 0.0f);
        }
        warps[warp] = sum32(lanes);
    }
    for (int warp = 4; warp < 32; ++warp) warps[warp] = 0.0f;
    return rsqrt32(fma32(sum32(warps), 1.0f / 128.0f, 1e-6f));
}

__device__ float rope32(float value, float partner, int index, int position) {
    float exponent = float(index % 32) * (1.0f / 32.0f);
    // The native source lowers reciprocal powf to a negative-exponent EX2.
    float inverse;
    float power = -mul32(exponent, 23.2534961700439453125f);
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(power));
    float frequency = mul32(float(position), inverse);
    float c = __cosf(frequency), s = __sinf(frequency);
    float cross = mul32(partner, s);
    return fma32(value, c, index < 32 ? -cross : cross);
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

__global__ void final_norm_projection(
    const at::BFloat16* weights, const at::BFloat16* input_hidden,
    const at::BFloat16* previous_history, int64_t position, int64_t prompt_length,
    float* scores, at::BFloat16* hidden_history) {
    __shared__ __align__(16) at::BFloat16 hidden[1024];
    __shared__ __align__(16) float activation[2048];
    for (int i = 0; i < 1024; ++i) hidden[i] = input_hidden[i];
    int64_t previous_count = position < prompt_length ? 0 : position - prompt_length + 1;
    for (int64_t i = 0; i < previous_count * 1024; ++i) hidden_history[i] = previous_history[i];
    if (position >= prompt_length - 1)
        for (int i = 0; i < 1024; ++i) hidden_history[previous_count * 1024 + i] = hidden[i];
    float rstd = hidden_rstd(hidden);
    for (int i = 0; i < 1024; ++i)
        activation[i] = mul32(mul32(float(hidden[i]), rstd), add32(1.0f, float(weights[254279680 + i])));

    for (int block = 0; block < 512; ++block) {
        for (int warp = 0; warp < 32; ++warp) {
            if (warp < 8)
                for (int row = block * 485 + warp; row < (block + 1) * 485; row += 8) {
                    float score = dot_float(weights, activation, row * 1024LL, 1024);
                    scores[row] = score;
                }
        }
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, int64_t input3, int64_t input4) {
    check_tensor(input0, at::kBFloat16, 254280704, "embedding and final norm weights");
    check_tensor(input1, at::kBFloat16, 1024, "hidden");
    check_position(input3, input4);
    int64_t previous_count = input3 < input4 ? 0 : input3 - input4 + 1;
    int64_t count = input3 < input4 - 1 ? 0 : input3 - input4 + 2;
    check_tensor(input2, at::kBFloat16, previous_count * 1024, "previous hidden history");
    TORCH_CHECK(input2.dim() == 2 && input2.size(0) == previous_count && input2.size(1) == 1024,
                "Invalid hidden history shape");
    auto scores = torch::empty({248320}, input0.options().dtype(at::kFloat));
    auto history = torch::empty({count, 1024}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    final_norm_projection<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
        input1.data_ptr<at::BFloat16>(), input2.data_ptr<at::BFloat16>(), input3, input4,
        scores.data_ptr<float>(), history.data_ptr<at::BFloat16>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, history};
}
