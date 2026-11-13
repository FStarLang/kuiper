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
            float chunk = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                chunk = fma32(float(weights[offset + k + j]), float(x[k + j]), chunk);
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
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum = fma32(float(weights[offset + k + j]), x[k + j], sum);
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
    return index < 32 ? fma32(value, c, -mul32(partner, s))
                      : fma32(partner, s, mul32(value, c));
}

__global__ void sequential_decode(
    const at::BFloat16* weights, const int64_t* prompt, int64_t steps, int64_t prompt_length,
    float* states, float* convolution, at::BFloat16* key_cache, at::BFloat16* value_cache,
    int64_t* tokens, at::BFloat16* hidden_states) {
    __shared__ __align__(16) at::BFloat16 hidden[1024], normalized[1024];
    __shared__ __align__(16) float projected[8192], activation[2048];
    float beta[16], alpha[16], q[128], k[128], v[128], lanes[32];
    float partial_output[4096], maxima[16], denominators[16];
    float block_values[512];
    int block_indices[512];
    for (int i = 0; i < 4718592; ++i) states[i] = 0.0f;
    for (int i = 0; i < 442368; ++i) convolution[i] = 0.0f;
    for (int i = 0; i < 6291456; ++i) {
        key_cache[i] = at::BFloat16(0.0f);
        value_cache[i] = at::BFloat16(0.0f);
    }
    int token = int(prompt[0]);
    for (int position = 0; position < prompt_length + steps - 1; ++position) {
        if (position < prompt_length) token = int(prompt[position]);
        for (int i = 0; i < 1024; ++i) hidden[i] = weights[int64_t(token) * 1024 + i];
        for (int layer = 0; layer < 24; ++layer) {
            bool full_attention = layer % 4 == 3;
            int64_t base = 254280704LL + (layer / 4) * 83018720LL + (layer % 4) * 21555360LL;
            int64_t extra = base + 11012096;
            float rstd = hidden_rstd(hidden);
            for (int i = 0; i < 1024; ++i)
                normalized[i] = at::BFloat16(mul32(mul32(float(hidden[i]), rstd),
                                                        add32(1.0f, float(weights[base + i]))));
            if (!full_attention) {
                for (int row = 0; row < 6144; ++row)
                    projected[row] = dot_bf16(weights, normalized, extra + row * 1024LL);
                for (int row = 0; row < 2048; ++row)
                    projected[6144 + row] = dot_bf16(weights, normalized, extra + 6291456 + row * 1024LL);
                for (int h = 0; h < 16; ++h) {
                    beta[h] = dot_bf16(weights, normalized, extra + 8388608 + h * 1024);
                    alpha[h] = dot_bf16(weights, normalized, extra + 8404992 + h * 1024);
                }
                int dn = (layer / 4) * 3 + layer % 4;
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
            } else {
                for (int row = 0; row < 4096; ++row)
                    projected[row] = dot_bf16(weights, normalized, extra + row * 1024LL);
                for (int row = 0; row < 512; ++row) {
                    projected[4096 + row] = dot_bf16(weights, normalized, extra + 4194304 + row * 1024LL);
                    projected[4608 + row] = dot_bf16(weights, normalized, extra + 4718592 + row * 1024LL);
                }
                for (int h = 0; h < 2; ++h) {
                    float scale = rsqrt32(fma32(strided_dot(projected + 4096 + h * 256,
                                                           projected + 4096 + h * 256, 256),
                                                 1.0f / 256.0f, 1e-6f));
                    for (int i = 0; i < 256; ++i) {
                        float normed = mul32(mul32(projected[4096 + h * 256 + i], scale),
                                             add32(1.0f, float(weights[extra + 5243136 + i])));
                        if (i < 64) {
                            int partner = i < 32 ? i + 32 : i - 32;
                            float pv = mul32(mul32(projected[4096 + h * 256 + partner], scale),
                                             add32(1.0f, float(weights[extra + 5243136 + partner])));
                            normed = rope32(normed, pv, i, position);
                        }
                        int offset = (layer / 4) * 1048576 + h * 524288 + position * 256 + i;
                        key_cache[offset] = at::BFloat16(normed);
                        value_cache[offset] = at::BFloat16(projected[4608 + h * 256 + i]);
                    }
                }
                for (int h = 0; h < 8; ++h) {
                    float scale = rsqrt32(fma32(strided_dot(projected + h * 512, projected + h * 512, 256),
                                                 1.0f / 256.0f, 1e-6f));
                    // Keep EX2 on the native runtime SFU path, not constant-folded after unrolling.
                    #pragma unroll 1
                    for (int i = 0; i < 32; ++i) {
                        float normed = mul32(mul32(projected[h * 512 + i], scale),
                                             add32(1.0f, float(weights[extra + 5242880 + i])));
                        float partner = mul32(mul32(projected[h * 512 + i + 32], scale),
                                              add32(1.0f, float(weights[extra + 5242912 + i])));
                        projected[h * 512 + i] = rope32(normed, partner, i, position);
                        projected[h * 512 + i + 32] = rope32(partner, normed, i + 32, position);
                    }
                    for (int i = 64; i < 256; ++i)
                        projected[h * 512 + i] = mul32(mul32(projected[h * 512 + i], scale),
                                                       add32(1.0f, float(weights[extra + 5242880 + i])));
                    for (int warp = 0; warp < 16; ++warp) {
                        float maximum = -INFINITY, denominator = 0.0f;
                        for (int i = 0; i < 256; ++i) partial_output[warp * 256 + i] = 0.0f;
                        for (int p = warp; p <= position; p += 16) {
                            int offset = (layer / 4) * 1048576 + (h / 4) * 524288 + p * 256;
                            for (int lane = 0; lane < 32; ++lane) {
                                float sum = 0.0f;
                                for (int e = 0; e < 8; ++e)
                                    sum = fma32(projected[h * 512 + lane * 8 + e],
                                                float(key_cache[offset + lane * 8 + e]), sum);
                                lanes[lane] = sum;
                            }
                            float score = mul32(sum32(lanes), 0.0625f);
                            float previous = maximum;
                            maximum = fmaxf(maximum, score);
                            float correction = exp32(add32(previous, -maximum));
                            float weight = exp32(add32(score, -maximum));
                            denominator = fma32(denominator, correction, weight);
                            for (int i = 0; i < 256; ++i)
                                partial_output[warp * 256 + i] = fma32(partial_output[warp * 256 + i],
                                    correction, mul32(weight, float(value_cache[offset + i])));
                        }
                        maxima[warp] = maximum;
                        denominators[warp] = denominator;
                    }
                    float maximum = -INFINITY;
                    for (int warp = 0; warp < 16; ++warp)
                        if (maxima[warp] > -INFINITY) maximum = fmaxf(maximum, maxima[warp]);
                    float total = 0.0f;
                    for (int i = 0; i < 256; ++i) activation[h * 256 + i] = 0.0f;
                    for (int warp = 0; warp < 16; ++warp) {
                        if (maxima[warp] > -INFINITY) {
                            float scale = exp32(add32(maxima[warp], -maximum));
                            total = fma32(denominators[warp], scale, total);
                            for (int i = 0; i < 256; ++i)
                                activation[h * 256 + i] = fma32(partial_output[warp * 256 + i], scale,
                                                                activation[h * 256 + i]);
                        }
                    }
                    float inverse = reciprocal32(total);
                    for (int i = 0; i < 256; ++i)
                        activation[h * 256 + i] = mul32(mul32(activation[h * 256 + i], inverse),
                                                        sigmoid32(projected[h * 512 + 256 + i]));
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
        }
        if (position >= prompt_length - 1)
            for (int i = 0; i < 1024; ++i)
                hidden_states[(position - prompt_length + 1) * 1024 + i] = hidden[i];
        float rstd = hidden_rstd(hidden);
        for (int i = 0; i < 1024; ++i)
            activation[i] = mul32(mul32(float(hidden[i]), rstd), add32(1.0f, float(weights[254279680 + i])));
        for (int block = 0; block < 512; ++block) {
            float values[32];
            int indices[32];
            for (int warp = 0; warp < 32; ++warp) {
                float maximum = -INFINITY;
                int index = -1;
                if (warp < 8)
                    for (int row = block * 485 + warp; row < (block + 1) * 485; row += 8) {
                        float score = dot_float(weights, activation, row * 1024LL, 1024);
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
        if (position >= prompt_length - 1) tokens[position - prompt_length + 1] = token;
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    auto states = torch::empty({18, 16, 128, 128}, input0.options().dtype(at::kFloat));
    auto convolution = torch::empty({18, 6144, 4}, input0.options().dtype(at::kFloat));
    auto key_cache = torch::empty({6, 2, 2048, 256}, input0.options());
    auto value_cache = torch::empty({6, 2, 2048, 256}, input0.options());
    auto tokens = torch::empty({input2}, input1.options());
    auto hidden_states = torch::empty({input2, 1024}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    sequential_decode<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(),
        input2, input1.numel(), states.data_ptr<float>(), convolution.data_ptr<float>(),
        key_cache.data_ptr<at::BFloat16>(), value_cache.data_ptr<at::BFloat16>(),
        tokens.data_ptr<int64_t>(), hidden_states.data_ptr<at::BFloat16>());
    return {tokens, hidden_states};
}
