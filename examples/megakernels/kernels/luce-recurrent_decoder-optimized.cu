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

__global__ void recurrent_decoder(
    const at::BFloat16* weights, const at::BFloat16* input_hidden,
    const float* input_states, const float* input_convolution, int64_t position,
    at::BFloat16* output_hidden, float* states, float* convolution) {
    __shared__ __align__(16) at::BFloat16 hidden[1024], normalized[1024];
    __shared__ __align__(16) float projected[8192], activation[2048];
    float beta[16], alpha[16], q[128], k[128], v[128], lanes[32];
    float partial_output[4096], maxima[16], denominators[16];

    for (int i = 0; i < 262144; ++i)
        states[i] = position == 0 ? 0.0f : input_states[i];
    for (int i = 0; i < 24576; ++i)
        convolution[i] = position == 0 ? 0.0f : input_convolution[i];
    for (int i = 0; i < 1024; ++i) hidden[i] = input_hidden[i];
    constexpr bool full_attention = false;
    int64_t base = 0;
    int64_t extra = base + 11012096;
    float rstd = hidden_rstd(hidden);
    for (int i = 0; i < 1024; ++i)
        normalized[i] = at::BFloat16(mul32(mul32(float(hidden[i]), rstd),
                                                add32(1.0f, float(weights[base + i]))));
    for (int row = 0; row < 6144; ++row)
        projected[row] = dot_bf16(weights, normalized, extra + row * 1024LL);
    for (int row = 0; row < 2048; ++row)
        projected[6144 + row] = dot_bf16(weights, normalized, extra + 6291456 + row * 1024LL);
    for (int h = 0; h < 16; ++h) {
        beta[h] = dot_bf16(weights, normalized, extra + 8388608 + h * 1024);
        alpha[h] = dot_bf16(weights, normalized, extra + 8404992 + h * 1024);
    }
    int dn = 0;
    for (int h = 0; h < 16; ++h) {
        for (int region = 0; region < 3; ++region) {
            for (int i = 0; i < 128; ++i) {
                int channel = region * 2048 + h * 128 + i;
                int offset = dn * 24576 + channel * 4;
                for (int t = 0; t < 3; ++t)
                    convolution[offset + t] = convolution[offset + t + 1];
                convolution[offset + 3] = projected[channel];
                float sum = 0.0f;
                for (int t = 0; t < 4; ++t)
                    sum = fma32(convolution[offset + t],
                                float(weights[extra + 8421376 + channel * 4 + t]), sum);
                float value = silu32(sum);
                if (region == 0) q[i] = value;
                if (region == 1) k[i] = value;
                if (region == 2) v[i] = value;
            }
        }
        float b = sigmoid32(beta[h]);
        float x = add32(alpha[h], float(weights[extra + 8445968 + h]));
        float sp = x > 20.0f ? x : __logf(add32(1.0f, exp32(x)));
        float decay = exp32(mul32(-exp32(float(weights[extra + 8445952 + h])), sp));
        float qscale = mul32(rsqrt32(add32(strided_dot(q, q, 128), 1e-6f)),
                             1.0f / 11.313708498984761f);
        float kscale = rsqrt32(add32(strided_dot(k, k, 128), 1e-6f));
        for (int i = 0; i < 128; ++i) {
            q[i] = mul32(q[i], qscale);
            k[i] = mul32(k[i], kscale);
        }
        float kq = strided_dot(k, q, 128);
        for (int j = 0; j < 128; ++j) {
            int offset = dn * 262144 + h * 16384 + j * 128;
            float stk = strided_dot(states + offset, k, 128);
            float sqv = strided_dot(states + offset, q, 128);
            float error = mul32(fma32(-decay, stk, v[j]), b);
            activation[h * 128 + j] = fma32(decay, sqv, mul32(error, kq));
            for (int i = 0; i < 128; ++i)
                states[offset + i] = fma32(states[offset + i], decay, mul32(k[i], error));
        }
        float scale = gated_rstd(activation + h * 128);
        for (int i = 0; i < 128; ++i) {
            float normed = mul32(mul32(activation[h * 128 + i], scale),
                                 float(weights[extra + 8445984 + i]));
            activation[h * 128 + i] = mul32(normed, silu32(projected[6144 + h * 128 + i]));
        }
    }
    int64_t out_weight = extra + (full_attention ? 5243392 : 8446112);
    for (int i = 0; i < 1024; ++i)
        hidden[i] = at::BFloat16(add32(dot_float(weights, activation, out_weight + i * 2048LL, 2048),
                                     float(hidden[i])));
    rstd = hidden_rstd(hidden);
    for (int i = 0; i < 1024; ++i)
        normalized[i] = at::BFloat16(mul32(mul32(float(hidden[i]), rstd),
                                                add32(1.0f, float(weights[base + 1024 + i]))));
    for (int row = 0; row < 3584; ++row) {
        float gate = dot_bf16(weights, normalized, base + 2048 + row * 1024LL);
        float up = dot_bf16(weights, normalized, base + 3672064 + row * 1024LL);
        projected[row] = mul32(silu32(gate), up);
    }
    for (int i = 0; i < 1024; ++i)
        hidden[i] = at::BFloat16(add32(dot_float(weights, projected, base + 7342080 + i * 3584LL, 3584),
                                     float(hidden[i])));

    for (int i = 0; i < 1024; ++i) output_hidden[i] = hidden[i];
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, torch::Tensor input3, int64_t input4) {
    check_tensor(input0, at::kBFloat16, 21555360, "recurrent layer weights");
    check_tensor(input1, at::kBFloat16, 1024, "hidden");
    TORCH_CHECK(input4 >= 0 && input4 < 2048, "Invalid recurrent position");
    if (input4 == 0) {
        check_empty(input2, "recurrent state");
        check_empty(input3, "convolution state");
    } else {
        check_tensor(input2, at::kFloat, 262144, "recurrent state");
        check_tensor(input3, at::kFloat, 24576, "convolution state");
    }
    auto hidden = torch::empty({1024}, input1.options());
    auto states = torch::empty({16, 128, 128}, input0.options().dtype(at::kFloat));
    auto convolution = torch::empty({6144, 4}, input0.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    recurrent_decoder<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
        input1.data_ptr<at::BFloat16>(), input4 == 0 ? nullptr : input2.data_ptr<float>(),
        input4 == 0 ? nullptr : input3.data_ptr<float>(), input4, hidden.data_ptr<at::BFloat16>(),
        states.data_ptr<float>(), convolution.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {hidden, states, convolution};
}
