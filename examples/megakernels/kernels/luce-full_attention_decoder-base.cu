#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
__device__ float rms_scale(const at::BFloat16* input, int offset, int width) {
    float partial[256];
    for (int i = 0; i < width / 4; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            float x = float(input[offset + 4 * i + j]);
            sum = __fadd_rn(sum, __fmul_rn(x, x));
        }
        partial[i] = sum;
    }
    for (int stride = width / 8; stride > 0; stride /= 2)
        for (int i = 0; i < stride; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + stride]);
    return rsqrtf(__fadd_rn(__fmul_rn(partial[0], 1.0f / width), 1.0e-6f));
}

__device__ float l2_scale(const at::BFloat16* input, int offset) {
    float partial[16];
    for (int i = 0; i < 16; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 8; ++j) {
            float x = float(input[offset + 8 * i + j]);
            sum = __fadd_rn(sum, float(at::BFloat16(__fmul_rn(x, x))));
        }
        partial[i] = sum;
    }
    for (int stride = 8; stride > 0; stride /= 2)
        for (int i = 0; i < stride; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + stride]);
    float sum = float(at::BFloat16(partial[0]));
    float shifted = float(at::BFloat16(__fadd_rn(sum, 1.0e-6f)));
    return float(at::BFloat16(rsqrtf(shifted)));
}

__device__ float recurrent_dot(const float* state, const float* vector, int offset, int column) {
    float partial[4];
    for (int y = 0; y < 4; ++y) {
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < 4; ++r)
            for (int i = y + 4 * r; i < 128; i += 16)
                sums[r] = __fadd_rn(sums[r], __fmul_rn(state[offset + i * 128 + column], vector[i]));
        partial[y] = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
    }
    return __fadd_rn(__fadd_rn(partial[0], partial[2]), __fadd_rn(partial[1], partial[3]));
}

__device__ void linear_attention_scalar(const at::BFloat16* weight, at::BFloat16* h,
                                        at::BFloat16* conv, float* state,
                                        int base, int state_layer) {
    int conv_offset = state_layer * 24576;
    for (int i = 0; i < 6144; ++i) {
        for (int j = 0; j < 3; ++j)
            conv[conv_offset + 4 * i + j] = conv[conv_offset + 4 * i + j + 1];
        conv[conv_offset + 4 * i + 3] = h[2048 + i];
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j)
            sum = __fmaf_rn(float(conv[conv_offset + 4 * i + j]),
                            float(weight[base + 19433472 + 4 * i + j]), sum);
        float x = float(at::BFloat16(sum));
        h[2048 + i] = at::BFloat16(x / __fadd_rn(1.0f, expf(-x)));
    }
    for (int head = 0; head < 16; ++head) {
        float query[128], key[128];
        float qscale = l2_scale(h, 2048 + head * 128);
        float kscale = l2_scale(h, 4096 + head * 128);
        for (int i = 0; i < 128; ++i) {
            float q = float(at::BFloat16(__fmul_rn(float(h[2048 + head * 128 + i]), qscale)));
            query[i] = __fmul_rn(q, 0.08838834764831845f);
            key[i] = float(at::BFloat16(__fmul_rn(float(h[4096 + head * 128 + i]), kscale)));
        }
        float b = float(h[10240 + head]);
        float beta = float(at::BFloat16(1.0f / __fadd_rn(1.0f, expf(-b))));
        float a = __fadd_rn(float(h[10256 + head]), float(weight[base + 19458064 + head]));
        float softplus = a > 20.0f ? a : log1pf(expf(a));
        float g = expf(__fmul_rn(-expf(float(weight[base + 19458048 + head])), softplus));
        int offset = state_layer * 262144 + head * 16384;
        for (int i = 0; i < 16384; ++i)
            state[offset + i] = __fmul_rn(state[offset + i], g);
        for (int j = 0; j < 128; ++j) {
            float memory = recurrent_dot(state, key, offset, j);
            float delta = __fmul_rn(__fsub_rn(float(h[6144 + head * 128 + j]), memory), beta);
            for (int i = 0; i < 128; ++i)
                state[offset + i * 128 + j] = __fadd_rn(state[offset + i * 128 + j],
                                                      __fmul_rn(key[i], delta));
            h[10400 + head * 128 + j] = at::BFloat16(recurrent_dot(state, query, offset, j));
        }
        float scale = rms_scale(h, 10400 + head * 128, 128);
        for (int i = 0; i < 128; ++i) {
            int index = head * 128 + i;
            float normed = float(at::BFloat16(__fmul_rn(float(h[10400 + index]), scale)));
            normed = float(at::BFloat16(__fmul_rn(normed, float(weight[base + 19458080 + i]))));
            float z = float(h[8192 + index]);
            float gate = z / __fadd_rn(1.0f, expf(-z));
            h[10400 + index] = at::BFloat16(__fmul_rn(normed, gate));
        }
    }
}

__device__ void attention_softmax(at::BFloat16* h, int offset, int length) {
    float value[2048], partial[32];
    float maximum = -INFINITY;
    int width = 1;
    while (width < length && width < 32) width *= 2;
    for (int i = 0; i < length; ++i) {
        float scaled = float(at::BFloat16(__fmul_rn(float(h[offset + i]), 0.0625f)));
        value[i] = __fadd_rn(scaled, 0.0f);
        maximum = maximum < value[i] ? value[i] : maximum;
    }
    for (int i = 0; i < width; ++i) {
        partial[i] = 0.0f;
        for (int j = i; j < length; j += width) {
            value[j] = expf(__fsub_rn(value[j], maximum));
            partial[i] = __fadd_rn(partial[i], value[j]);
        }
    }
    for (int stride = width / 2; stride > 0; stride /= 2)
        for (int i = 0; i < stride; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + stride]);
    for (int i = 0; i < length; ++i)
        h[offset + i] = at::BFloat16(value[i] / partial[0]);
}

__device__ void apply_rms(const at::BFloat16* weight, at::BFloat16* h, int source,
                          int destination, int width, int rows, int stride, int weight_offset) {
    for (int row = 0; row < rows; ++row) {
        float scale = rms_scale(h, source + row * stride, width);
        for (int i = 0; i < width; ++i) {
            float x = __fmul_rn(float(h[source + row * stride + i]), scale);
            h[destination + row * width + i] = at::BFloat16(
                __fmul_rn(x, __fadd_rn(1.0f, float(weight[weight_offset + i]))));
        }
    }
}

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

__device__ void prepare_attention(const at::BFloat16* weight, at::BFloat16* h, int base) {
    for (int head = 0; head < 8; ++head)
        for (int i = 0; i < 256; ++i)
            h[8192 + head * 256 + i] = h[2048 + head * 512 + 256 + i];
    apply_rms(weight, h, 2048, 2048, 256, 8, 512, base + 16254976);
    apply_rms(weight, h, 6144, 6144, 256, 2, 256, base + 16255232);
    for (int head = 0; head < 10; ++head) {
        int offset = head < 8 ? 2048 + head * 256 : 6144 + (head - 8) * 256;
        for (int i = 0; i < 32; ++i) {
            float x = float(h[offset + i]), y = float(h[offset + i + 32]);
            float cosine = float(h[15600 + i]), sine = float(h[15632 + i]);
            float xc = float(at::BFloat16(__fmul_rn(x, cosine)));
            float ys = float(at::BFloat16(__fmul_rn(-y, sine)));
            float yc = float(at::BFloat16(__fmul_rn(y, cosine)));
            float xs = float(at::BFloat16(__fmul_rn(x, sine)));
            h[offset + i] = at::BFloat16(__fadd_rn(xc, ys));
            h[offset + i + 32] = at::BFloat16(__fadd_rn(yc, xs));
        }
    }
}

__device__ void attention_simt(const at::BFloat16* cache, at::BFloat16* h, int layer,
                               int head, int length, int matrix, int input_offset, int output_offset) {
    bool score = matrix == 1;
    bool linear_finish = score && ((length >= 388 && length <= 415) || length >= 444);
    int groups = score ? (length >= 57 && length <= 139 ? 32 : 16) : (length < 426 ? 32 : 16);
    int columns = score ? 256 : length;
    int rows = score ? length : 256;
    bool interleaved = score && length >= 417 && length <= 443;
    int chunk = linear_finish || interleaved ? 1 : (score ? 256 / groups : (length < 426 ? 4 : (length + 15) / 16));
    float partial[32];
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            float sum = 0.0f;
            for (int start = group * chunk; start < columns; start += groups * chunk)
                for (int i = start; i < start + chunk && i < columns; ++i) {
                    int index = score ? ((layer / 4) * 4096 + row) * 512 + (head / 4) * 256 + i
                                      : ((layer / 4) * 4096 + 2048 + i) * 512 + (head / 4) * 256 + row;
                    sum = __fmaf_rn(float(cache[index]), float(h[input_offset + i]), sum);
                }
            partial[group] = sum;
        }
        if (linear_finish) {
            for (int i = 1; i < groups; ++i) partial[0] = __fadd_rn(partial[0], partial[i]);
        } else {
            for (int stride = groups / 2; stride > 0; stride /= 2)
                for (int i = 0; i < stride; ++i)
                    partial[i] = __fadd_rn(partial[i], partial[i + stride]);
        }
        h[output_offset + row] = at::BFloat16(partial[0]);
    }
}

__device__ void decoder_layer(const at::BFloat16* weight, at::BFloat16* h,
                              at::BFloat16* conv, float* state, at::BFloat16* cache,
                              bool linear, int position) {
    int lane = threadIdx.x, group = lane / 4, pair = lane % 4;
    int layer = linear ? 0 : 3;
    int base = 0;
    int phases = linear ? 8 : 23;
    for (int phase = 0; phase < phases; ++phase) {
        bool attention = !linear && phase >= 3 && phase < 19;
        int kind = !linear && phase >= 19 ? phase - 15 : phase;
        int head = (phase - 3) / 2;
        if (lane == 0) {
            if (phase == 0) {
                apply_rms(weight, h, 0, 1024, 1024, 1, 1024, base);
            }
            if (linear && phase == 4)
                linear_attention_scalar(weight, h, conv, state, base, layer - layer / 4);
            if (!linear && phase == 3) {
                prepare_attention(weight, h, base);
                for (int i = 0; i < 512; ++i) {
                    cache[((layer / 4) * 4096 + position) * 512 + i] = h[6144 + i];
                    cache[((layer / 4) * 4096 + 2048 + position) * 512 + i] = h[6656 + i];
                }
            }
            if (attention && phase % 2 == 0)
                attention_softmax(h, 13500, position + 1);
            if (!linear && phase == 19)
                for (int i = 0; i < 2048; ++i) {
                    float gate = float(at::BFloat16(1.0f / __fadd_rn(1.0f, expf(-float(h[8192 + i])))));
                    h[10400 + i] = at::BFloat16(__fmul_rn(float(h[10400 + i]), gate));
                }
            if (!attention && kind == 5) {
                for (int i = 0; i < 1024; ++i)
                    h[i] = at::BFloat16(__fadd_rn(float(h[i]), float(h[12500 + i])));
                apply_rms(weight, h, 0, 1024, 1024, 1, 1024, base + 1024);
            }
            if (!attention && kind == 7)
                for (int i = 0; i < 3584; ++i) {
                    float x = float(h[2048 + i]);
                    float gate = float(at::BFloat16(x / __fadd_rn(1.0f, expf(-x))));
                    h[2048 + i] = at::BFloat16(__fmul_rn(gate, float(h[5632 + i])));
                }
        }
        __syncwarp();
        int rows = 0, columns = 1024, input_offset = 1024, output_offset = 12500;
        int weight_offset = 0, matrix = 0, splits = 1;
        if (attention) {
            matrix = phase % 2 == 1 ? 1 : 2;
            rows = matrix == 1 ? position + 1 : 256;
            columns = matrix == 1 ? 256 : position + 1;
            input_offset = matrix == 1 ? 2048 + head * 256 : 13500;
            output_offset = matrix == 1 ? 13500 : 10400 + head * 256;
        } else if (kind == 0) {
            rows = linear ? 6144 : 4096;
            weight_offset = base + 11012096;
            output_offset = 2048;
        } else if (kind == 1) {
            rows = linear ? 2048 : 512;
            weight_offset = base + (linear ? 17303552 : 15206400);
            output_offset = linear ? 8192 : 6144;
        } else if (kind == 2) {
            rows = linear ? 16 : 512;
            weight_offset = base + (linear ? 19400704 : 15730688);
            output_offset = linear ? 10240 : 6656;
        } else if (kind == 3) {
            rows = 16;
            weight_offset = base + 19417088;
            output_offset = 10256;
        } else if (kind == 4) {
            rows = 1024;
            columns = 2048;
            input_offset = 10400;
            weight_offset = base + (linear ? 19458208 : 16255488);
        } else if (kind == 5 || kind == 6) {
            rows = 3584;
            weight_offset = base + (kind == 5 ? 2048 : 3672064);
            output_offset = kind == 5 ? 2048 : 5632;
        } else if (kind == 7) {
            rows = 1024;
            columns = 3584;
            input_offset = 2048;
            weight_offset = base + 7342080;
            splits = 2;
        }
        if (attention && (position + 1) % 8 != 0) {
            if (lane == 0)
                attention_simt(cache, h, layer, head, position + 1, matrix, input_offset, output_offset);
        } else {
        for (int row = 0; row < rows; row += 16) {
            float total0 = 0.0f, total2 = 0.0f;
            for (int part = 0; part < splits; ++part) {
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                for (int k = part * (columns / splits); k < (part + 1) * (columns / splits); k += 16) {
                    uint32_t a[4], b[2];
                    for (int r = 0; r < 4; ++r) {
                        int m = row + group + (r % 2) * 8;
                        int n = k + pair * 2 + (r / 2) * 8;
                        a[r] = 0;
                        for (int u = 0; u < 2; ++u)
                            if (m < rows && n + u < columns) {
                                at::BFloat16 value;
                                if (matrix == 0)
                                    value = weight[weight_offset + m * columns + n + u];
                                else if (matrix == 1)
                                    value = cache[((layer / 4) * 4096 + m) * 512 + (head / 4) * 256 + n + u];
                                else
                                    value = cache[((layer / 4) * 4096 + 2048 + n + u) * 512 + (head / 4) * 256 + m];
                                a[r] |= uint32_t(value.x) << (16 * u);
                            }
                    }
                    for (int r = 0; r < 2; ++r) {
                        int n = k + pair * 2 + r * 8;
                        b[r] = 0;
                        for (int u = 0; u < 2; ++u)
                            if (n + u < columns)
                                b[r] |= uint32_t(h[input_offset + n + u].x) << (16 * u);
                    }
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
                }
                total0 = __fadd_rn(total0, d0);
                total2 = __fadd_rn(total2, d2);
            }
            if (pair == 0) {
                int out = output_offset + row;
                if (row + group < rows) h[out + group] = at::BFloat16(total0);
                if (row + group + 8 < rows) h[out + group + 8] = at::BFloat16(total2);
            }
        
        }
        }
        __syncwarp();
        if (lane == 0 && !attention && kind == 7)
            for (int i = 0; i < 1024; ++i)
                h[i] = at::BFloat16(__fadd_rn(float(h[i]), float(h[12500 + i])));
        __syncwarp();
    }
}

void check_tensor(const torch::Tensor& value, const torch::Tensor& owner,
                  at::ScalarType dtype, int64_t count, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.numel() == count, name, ": invalid device, layout, dtype or size");
    TORCH_CHECK(value.device() == owner.device(), name, ": tensors must share a CUDA device");
}

__global__ void full_attention_kernel(const at::BFloat16* weight, const at::BFloat16* input,
                                     const at::BFloat16* rope, const at::BFloat16* previous_keys,
                                     const at::BFloat16* previous_values, at::BFloat16* cache,
                                     at::BFloat16* output, at::BFloat16* keys,
                                     at::BFloat16* values, int position) {
    __shared__ at::BFloat16 h[16000];
    int lane = threadIdx.x;
    if (lane == 0) {
        for (int i = 0; i < 1024; ++i) h[i] = input[i];
        for (int i = 0; i < 64; ++i) h[15600 + i] = rope[i];
        for (int i = 0; i < position * 512; ++i) {
            cache[i] = previous_keys[i];
            cache[2048 * 512 + i] = previous_values[i];
        }
    }
    __syncwarp();
    decoder_layer(weight, h, nullptr, nullptr, cache, false, position);
    if (lane == 0) {
        for (int i = 0; i < 1024; ++i) output[i] = h[i];
        for (int i = 0; i < (position + 1) * 512; ++i) {
            keys[i] = cache[i];
            values[i] = cache[2048 * 512 + i];
        }
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, torch::Tensor input3,
                                    torch::Tensor input4, int64_t input5) {
    TORCH_CHECK(input5 >= 0 && input5 < 2048, "invalid attention position");
    check_tensor(input0, input0, at::kBFloat16, 18352640, "full-attention layer weights");
    check_tensor(input1, input0, at::kBFloat16, 1024, "hidden");
    check_tensor(input2, input0, at::kBFloat16, 64, "RoPE cosines and sines");
    check_tensor(input3, input0, at::kBFloat16, input5 * 512, "live key prefix");
    check_tensor(input4, input0, at::kBFloat16, input5 * 512, "live value prefix");
    c10::cuda::CUDAGuard guard(input0.device());
    auto output = torch::empty({1024}, input0.options());
    auto keys = torch::empty({input5 + 1, 512}, input0.options());
    auto values = torch::empty({input5 + 1, 512}, input0.options());
    auto cache = torch::empty({4096 * 512}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    full_attention_kernel<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
        input1.data_ptr<at::BFloat16>(), input2.data_ptr<at::BFloat16>(), input3.data_ptr<at::BFloat16>(),
        input4.data_ptr<at::BFloat16>(), cache.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(),
        keys.data_ptr<at::BFloat16>(), values.data_ptr<at::BFloat16>(), input5);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, keys, values};
}
