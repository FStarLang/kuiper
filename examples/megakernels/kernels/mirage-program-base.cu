#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

__global__ void embedding(const at::BFloat16* weights, const int64_t* tokens,
                          at::BFloat16* hidden, at::BFloat16* rotary) {
    for (int row = 0; row < 7; ++row)
        for (int col = 0; col < 1024; ++col)
            hidden[row * 1024 + col] = weights[tokens[row] * 1024 + col];
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 128; ++col) {
            float power = powf(1000000.0f, float(col % 64) / 64.0f);
            float angle = __fmul_rn(float(row), 1.0f / power);
            rotary[row * 128 + col] = at::BFloat16(cosf(angle));
            rotary[896 + row * 128 + col] = at::BFloat16(sinf(angle));
        }
    }
}

__global__ void rmsnorm(const at::BFloat16* x, const at::BFloat16* weights,
                        at::BFloat16* output, int offset) {
    float partial[128];
    for (int row = 0; row < 7; ++row) {
        for (int lane = 0; lane < 128; ++lane) {
            float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
            for (int col = lane * 4; col < 1024; col += 512) {
                float a = float(x[row * 1024 + col]);
                float b = float(x[row * 1024 + col + 1]);
                float c = float(x[row * 1024 + col + 2]);
                float d = float(x[row * 1024 + col + 3]);
                v0 = __fadd_rn(v0, __fmul_rn(a, a));
                v1 = __fadd_rn(v1, __fmul_rn(b, b));
                v2 = __fadd_rn(v2, __fmul_rn(c, c));
                v3 = __fadd_rn(v3, __fmul_rn(d, d));
            }
            partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(v0, v1), v2), v3);
        }
        for (int stride = 64; stride > 0; stride /= 2)
            for (int lane = 0; lane < stride; ++lane)
                partial[lane] = __fadd_rn(partial[lane], partial[lane + stride]);
        float scale = rsqrtf(__fadd_rn(__fmul_rn(partial[0], 1.0f / 1024), 1.0e-6f));
        for (int col = 0; col < 1024; ++col) {
            at::BFloat16 normalized = at::BFloat16(__fmul_rn(float(x[row * 1024 + col]), scale));
            output[row * 1024 + col] = at::BFloat16(__fmul_rn(float(normalized), float(weights[offset + col])));
        }
    }
}

__global__ void linear(const at::BFloat16* x, const at::BFloat16* weights,
                       at::BFloat16* output, int columns, int width,
                       int offset, int splits) {
    int lane = threadIdx.x;
    int group = lane / 4;
    int pair = (lane % 4) * 2;
    for (int tile = 0; tile < columns; tile += 16) {
        float total0 = 0, total1 = 0, total2 = 0, total3 = 0;
        for (int split = 0; split < splits; ++split) {
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int k = split * (width / splits); k < (split + 1) * (width / splits); k += 16) {
                int a = offset + (tile + group) * width + k + pair;
                uint32_t a0 = uint32_t(weights[a].x) | (uint32_t(weights[a + 1].x) << 16);
                uint32_t a1 = uint32_t(weights[a + 8 * width].x) | (uint32_t(weights[a + 8 * width + 1].x) << 16);
                uint32_t a2 = uint32_t(weights[a + 8].x) | (uint32_t(weights[a + 9].x) << 16);
                uint32_t a3 = uint32_t(weights[a + 8 * width + 8].x) | (uint32_t(weights[a + 8 * width + 9].x) << 16);
                uint32_t b0 = 0, b1 = 0;
                if (group < 7) {
                    int b = group * width + k + pair;
                    b0 = uint32_t(x[b].x) | (uint32_t(x[b + 1].x) << 16);
                    b1 = uint32_t(x[b + 8].x) | (uint32_t(x[b + 9].x) << 16);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            total0 = split == 0 ? c0 : __fadd_rn(total0, c0);
            total1 = split == 0 ? c1 : __fadd_rn(total1, c1);
            total2 = split == 0 ? c2 : __fadd_rn(total2, c2);
            total3 = split == 0 ? c3 : __fadd_rn(total3, c3);
        }
        output[pair * columns + tile + group] = at::BFloat16(total0);
        output[pair * columns + tile + group + 8] = at::BFloat16(total2);
        if (pair + 1 < 7) {
            output[(pair + 1) * columns + tile + group] = at::BFloat16(total1);
            output[(pair + 1) * columns + tile + group + 8] = at::BFloat16(total3);
        }
    }
}

__global__ void attention(const at::BFloat16* qkv, const at::BFloat16* weights,
                         const at::BFloat16* rotary, at::BFloat16* output, int offset) {
    at::BFloat16 qk[7 * 3072];
    float partial[32];
    at::BFloat16 normalized[128];
    for (int row = 0; row < 7; ++row) {
        for (int head = 0; head < 24; ++head) {
            int start = row * 4096 + head * 128;
            for (int lane = 0; lane < 32; ++lane) {
                float a = float(qkv[start + lane * 4]);
                float b = float(qkv[start + lane * 4 + 1]);
                float c = float(qkv[start + lane * 4 + 2]);
                float d = float(qkv[start + lane * 4 + 3]);
                partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(
                    __fmul_rn(a, a), __fmul_rn(b, b)), __fmul_rn(c, c)), __fmul_rn(d, d));
            }
            for (int stride = 16; stride > 0; stride /= 2)
                for (int lane = 0; lane < stride; ++lane)
                    partial[lane] = __fadd_rn(partial[lane], partial[lane + stride]);
            float scale = rsqrtf(__fadd_rn(__fmul_rn(partial[0], 1.0f / 128), 1.0e-6f));
            for (int col = 0; col < 128; ++col) {
                at::BFloat16 value = at::BFloat16(__fmul_rn(float(qkv[start + col]), scale));
                normalized[col] = at::BFloat16(__fmul_rn(
                    float(value), float(weights[offset + (head < 16 ? 0 : 128) + col])));
            }
            for (int col = 0; col < 128; ++col) {
                float rotated = float(normalized[(col + 64) % 128]);
                if (col < 64) rotated = -rotated;
                at::BFloat16 a = at::BFloat16(__fmul_rn(float(normalized[col]), float(rotary[row * 128 + col])));
                at::BFloat16 b = at::BFloat16(__fmul_rn(rotated, float(rotary[896 + row * 128 + col])));
                qk[row * 3072 + head * 128 + col] = at::BFloat16(__fadd_rn(float(a), float(b)));
            }
        }
    }
    float scores[8];
    float exponentials[8];
    at::BFloat16 probabilities[7];
    for (int head = 0; head < 16; ++head) {
        for (int row = 0; row < 7; ++row) {
            for (int key = 0; key <= row; ++key) {
                for (int lane = 0; lane < 16; ++lane) {
                    float sum = 0.0f;
                    for (int col = lane; col < 128; col += 16)
                        sum = __fmaf_rn(float(qk[row * 3072 + head * 128 + col]),
                            float(qk[key * 3072 + 2048 + (head / 2) * 128 + col]), sum);
                    partial[lane] = sum;
                }
                for (int stride = 8; stride > 0; stride /= 2)
                    for (int lane = 0; lane < stride; ++lane)
                        partial[lane] = __fadd_rn(partial[lane], partial[lane + stride]);
                at::BFloat16 dot = at::BFloat16(partial[0]);
                at::BFloat16 scaled = at::BFloat16(__fmul_rn(float(dot), 0.08838834764831845f));
                scores[key] = float(scaled);
            }
            float maximum = scores[0];
            for (int key = 1; key <= row; ++key) maximum = fmaxf(maximum, scores[key]);
            for (int key = 0; key < 8; ++key) {
                exponentials[key] = key <= row ? expf(__fsub_rn(scores[key], maximum)) : 0.0f;
                partial[key] = exponentials[key];
            }
            for (int stride = 4; stride > 0; stride /= 2)
                for (int key = 0; key < stride; ++key)
                    partial[key] = __fadd_rn(partial[key], partial[key + stride]);
            for (int key = 0; key <= row; ++key)
                probabilities[key] = at::BFloat16(exponentials[key] / partial[0]);
            for (int col = 0; col < 128; ++col) {
                float sum = 0.0f;
                for (int key = 0; key <= row; ++key)
                    sum = __fmaf_rn(float(probabilities[key]),
                        float(qkv[key * 4096 + 3072 + (head / 2) * 128 + col]), sum);
                output[row * 2048 + head * 128 + col] = at::BFloat16(sum);
            }
        }
    }
}

__global__ void add_norm(const at::BFloat16* residual, const at::BFloat16* delta,
                        const at::BFloat16* weights, at::BFloat16* combined,
                        at::BFloat16* normalized, int offset) {
    float values[1024];
    float partial[128];
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 1024; ++col) {
            at::BFloat16 value = at::BFloat16(__fadd_rn(
                float(residual[row * 1024 + col]), float(delta[row * 1024 + col])));
            combined[row * 1024 + col] = value;
            values[col] = float(value);
        }
        for (int lane = 0; lane < 128; ++lane) {
            float sums[4] = {0, 0, 0, 0};
            for (int col = lane * 4; col < 1024; col += 512)
                for (int j = 0; j < 4; ++j)
                    sums[j] = __fadd_rn(sums[j], __fmul_rn(values[col + j], values[col + j]));
            partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
        }
        for (int stride = 64; stride > 0; stride /= 2)
            for (int lane = 0; lane < stride; ++lane)
                partial[lane] = __fadd_rn(partial[lane], partial[lane + stride]);
        float scale = rsqrtf(__fadd_rn(__fmul_rn(partial[0], 1.0f / 1024), 1.0e-6f));
        for (int col = 0; col < 1024; ++col) {
            at::BFloat16 value = at::BFloat16(__fmul_rn(values[col], scale));
            normalized[row * 1024 + col] = at::BFloat16(__fmul_rn(float(value), float(weights[offset + col])));
        }
    }
}

__global__ void silu_product(const at::BFloat16* gate_up, at::BFloat16* output) {
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 3072; ++col) {
            float gate = float(gate_up[row * 6144 + col]);
            at::BFloat16 activated = at::BFloat16(gate / __fadd_rn(1.0f, expf(-gate)));
            output[row * 3072 + col] = at::BFloat16(__fmul_rn(
                float(activated), float(gate_up[row * 6144 + 3072 + col])));
        }
    }
}

__global__ void residual_add(const at::BFloat16* residual, const at::BFloat16* delta,
                            at::BFloat16* output) {
    for (int i = 0; i < 7 * 1024; ++i)
        output[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(delta[i])));
}

__global__ void select_token(const at::BFloat16* logits, const int64_t* tokens,
                            int64_t* output) {
    for (int i = 0; i < 7; ++i) output[i] = tokens[i];
    float best = float(logits[6 * 151936]);
    int index = 0;
    for (int i = 1; i < 151936; ++i) {
        float value = float(logits[6 * 151936 + i]);
        if (!isnan(best) && (isnan(value) || value > best)) {
            best = value;
            index = i;
        }
    }
    output[7] = index;
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1, int64_t input2) {
    auto hidden0 = torch::empty({7, 1024}, input0.options());
    auto rotary = torch::empty({2, 7, 128}, input0.options());
    auto norm0 = torch::empty({7, 1024}, input0.options());
    auto qkv0 = torch::empty({7, 4096}, input0.options());
    auto attn0 = torch::empty({7, 2048}, input0.options());
    auto projected0 = torch::empty({7, 1024}, input0.options());
    auto residual0 = torch::empty({7, 1024}, input0.options());
    auto postnorm0 = torch::empty({7, 1024}, input0.options());
    auto gateup0 = torch::empty({7, 6144}, input0.options());
    auto activated0 = torch::empty({7, 3072}, input0.options());
    auto down0 = torch::empty({7, 1024}, input0.options());
    auto hidden1 = torch::empty({7, 1024}, input0.options());
    auto norm1 = torch::empty({7, 1024}, input0.options());
    auto qkv1 = torch::empty({7, 4096}, input0.options());
    auto attn1 = torch::empty({7, 2048}, input0.options());
    auto projected1 = torch::empty({7, 1024}, input0.options());
    auto residual1 = torch::empty({7, 1024}, input0.options());
    auto postnorm1 = torch::empty({7, 1024}, input0.options());
    auto gateup1 = torch::empty({7, 6144}, input0.options());
    auto activated1 = torch::empty({7, 3072}, input0.options());
    auto down1 = torch::empty({7, 1024}, input0.options());
    auto hidden2 = torch::empty({7, 1024}, input0.options());
    auto norm2 = torch::empty({7, 1024}, input0.options());
    auto qkv2 = torch::empty({7, 4096}, input0.options());
    auto attn2 = torch::empty({7, 2048}, input0.options());
    auto projected2 = torch::empty({7, 1024}, input0.options());
    auto residual2 = torch::empty({7, 1024}, input0.options());
    auto postnorm2 = torch::empty({7, 1024}, input0.options());
    auto gateup2 = torch::empty({7, 6144}, input0.options());
    auto activated2 = torch::empty({7, 3072}, input0.options());
    auto down2 = torch::empty({7, 1024}, input0.options());
    auto hidden3 = torch::empty({7, 1024}, input0.options());
    auto norm3 = torch::empty({7, 1024}, input0.options());
    auto qkv3 = torch::empty({7, 4096}, input0.options());
    auto attn3 = torch::empty({7, 2048}, input0.options());
    auto projected3 = torch::empty({7, 1024}, input0.options());
    auto residual3 = torch::empty({7, 1024}, input0.options());
    auto postnorm3 = torch::empty({7, 1024}, input0.options());
    auto gateup3 = torch::empty({7, 6144}, input0.options());
    auto activated3 = torch::empty({7, 3072}, input0.options());
    auto down3 = torch::empty({7, 1024}, input0.options());
    auto hidden4 = torch::empty({7, 1024}, input0.options());
    auto norm4 = torch::empty({7, 1024}, input0.options());
    auto qkv4 = torch::empty({7, 4096}, input0.options());
    auto attn4 = torch::empty({7, 2048}, input0.options());
    auto projected4 = torch::empty({7, 1024}, input0.options());
    auto residual4 = torch::empty({7, 1024}, input0.options());
    auto postnorm4 = torch::empty({7, 1024}, input0.options());
    auto gateup4 = torch::empty({7, 6144}, input0.options());
    auto activated4 = torch::empty({7, 3072}, input0.options());
    auto down4 = torch::empty({7, 1024}, input0.options());
    auto hidden5 = torch::empty({7, 1024}, input0.options());
    auto norm5 = torch::empty({7, 1024}, input0.options());
    auto qkv5 = torch::empty({7, 4096}, input0.options());
    auto attn5 = torch::empty({7, 2048}, input0.options());
    auto projected5 = torch::empty({7, 1024}, input0.options());
    auto residual5 = torch::empty({7, 1024}, input0.options());
    auto postnorm5 = torch::empty({7, 1024}, input0.options());
    auto gateup5 = torch::empty({7, 6144}, input0.options());
    auto activated5 = torch::empty({7, 3072}, input0.options());
    auto down5 = torch::empty({7, 1024}, input0.options());
    auto hidden6 = torch::empty({7, 1024}, input0.options());
    auto norm6 = torch::empty({7, 1024}, input0.options());
    auto qkv6 = torch::empty({7, 4096}, input0.options());
    auto attn6 = torch::empty({7, 2048}, input0.options());
    auto projected6 = torch::empty({7, 1024}, input0.options());
    auto residual6 = torch::empty({7, 1024}, input0.options());
    auto postnorm6 = torch::empty({7, 1024}, input0.options());
    auto gateup6 = torch::empty({7, 6144}, input0.options());
    auto activated6 = torch::empty({7, 3072}, input0.options());
    auto down6 = torch::empty({7, 1024}, input0.options());
    auto hidden7 = torch::empty({7, 1024}, input0.options());
    auto norm7 = torch::empty({7, 1024}, input0.options());
    auto qkv7 = torch::empty({7, 4096}, input0.options());
    auto attn7 = torch::empty({7, 2048}, input0.options());
    auto projected7 = torch::empty({7, 1024}, input0.options());
    auto residual7 = torch::empty({7, 1024}, input0.options());
    auto postnorm7 = torch::empty({7, 1024}, input0.options());
    auto gateup7 = torch::empty({7, 6144}, input0.options());
    auto activated7 = torch::empty({7, 3072}, input0.options());
    auto down7 = torch::empty({7, 1024}, input0.options());
    auto hidden8 = torch::empty({7, 1024}, input0.options());
    auto norm8 = torch::empty({7, 1024}, input0.options());
    auto qkv8 = torch::empty({7, 4096}, input0.options());
    auto attn8 = torch::empty({7, 2048}, input0.options());
    auto projected8 = torch::empty({7, 1024}, input0.options());
    auto residual8 = torch::empty({7, 1024}, input0.options());
    auto postnorm8 = torch::empty({7, 1024}, input0.options());
    auto gateup8 = torch::empty({7, 6144}, input0.options());
    auto activated8 = torch::empty({7, 3072}, input0.options());
    auto down8 = torch::empty({7, 1024}, input0.options());
    auto hidden9 = torch::empty({7, 1024}, input0.options());
    auto norm9 = torch::empty({7, 1024}, input0.options());
    auto qkv9 = torch::empty({7, 4096}, input0.options());
    auto attn9 = torch::empty({7, 2048}, input0.options());
    auto projected9 = torch::empty({7, 1024}, input0.options());
    auto residual9 = torch::empty({7, 1024}, input0.options());
    auto postnorm9 = torch::empty({7, 1024}, input0.options());
    auto gateup9 = torch::empty({7, 6144}, input0.options());
    auto activated9 = torch::empty({7, 3072}, input0.options());
    auto down9 = torch::empty({7, 1024}, input0.options());
    auto hidden10 = torch::empty({7, 1024}, input0.options());
    auto norm10 = torch::empty({7, 1024}, input0.options());
    auto qkv10 = torch::empty({7, 4096}, input0.options());
    auto attn10 = torch::empty({7, 2048}, input0.options());
    auto projected10 = torch::empty({7, 1024}, input0.options());
    auto residual10 = torch::empty({7, 1024}, input0.options());
    auto postnorm10 = torch::empty({7, 1024}, input0.options());
    auto gateup10 = torch::empty({7, 6144}, input0.options());
    auto activated10 = torch::empty({7, 3072}, input0.options());
    auto down10 = torch::empty({7, 1024}, input0.options());
    auto hidden11 = torch::empty({7, 1024}, input0.options());
    auto norm11 = torch::empty({7, 1024}, input0.options());
    auto qkv11 = torch::empty({7, 4096}, input0.options());
    auto attn11 = torch::empty({7, 2048}, input0.options());
    auto projected11 = torch::empty({7, 1024}, input0.options());
    auto residual11 = torch::empty({7, 1024}, input0.options());
    auto postnorm11 = torch::empty({7, 1024}, input0.options());
    auto gateup11 = torch::empty({7, 6144}, input0.options());
    auto activated11 = torch::empty({7, 3072}, input0.options());
    auto down11 = torch::empty({7, 1024}, input0.options());
    auto hidden12 = torch::empty({7, 1024}, input0.options());
    auto norm12 = torch::empty({7, 1024}, input0.options());
    auto qkv12 = torch::empty({7, 4096}, input0.options());
    auto attn12 = torch::empty({7, 2048}, input0.options());
    auto projected12 = torch::empty({7, 1024}, input0.options());
    auto residual12 = torch::empty({7, 1024}, input0.options());
    auto postnorm12 = torch::empty({7, 1024}, input0.options());
    auto gateup12 = torch::empty({7, 6144}, input0.options());
    auto activated12 = torch::empty({7, 3072}, input0.options());
    auto down12 = torch::empty({7, 1024}, input0.options());
    auto hidden13 = torch::empty({7, 1024}, input0.options());
    auto norm13 = torch::empty({7, 1024}, input0.options());
    auto qkv13 = torch::empty({7, 4096}, input0.options());
    auto attn13 = torch::empty({7, 2048}, input0.options());
    auto projected13 = torch::empty({7, 1024}, input0.options());
    auto residual13 = torch::empty({7, 1024}, input0.options());
    auto postnorm13 = torch::empty({7, 1024}, input0.options());
    auto gateup13 = torch::empty({7, 6144}, input0.options());
    auto activated13 = torch::empty({7, 3072}, input0.options());
    auto down13 = torch::empty({7, 1024}, input0.options());
    auto hidden14 = torch::empty({7, 1024}, input0.options());
    auto norm14 = torch::empty({7, 1024}, input0.options());
    auto qkv14 = torch::empty({7, 4096}, input0.options());
    auto attn14 = torch::empty({7, 2048}, input0.options());
    auto projected14 = torch::empty({7, 1024}, input0.options());
    auto residual14 = torch::empty({7, 1024}, input0.options());
    auto postnorm14 = torch::empty({7, 1024}, input0.options());
    auto gateup14 = torch::empty({7, 6144}, input0.options());
    auto activated14 = torch::empty({7, 3072}, input0.options());
    auto down14 = torch::empty({7, 1024}, input0.options());
    auto hidden15 = torch::empty({7, 1024}, input0.options());
    auto norm15 = torch::empty({7, 1024}, input0.options());
    auto qkv15 = torch::empty({7, 4096}, input0.options());
    auto attn15 = torch::empty({7, 2048}, input0.options());
    auto projected15 = torch::empty({7, 1024}, input0.options());
    auto residual15 = torch::empty({7, 1024}, input0.options());
    auto postnorm15 = torch::empty({7, 1024}, input0.options());
    auto gateup15 = torch::empty({7, 6144}, input0.options());
    auto activated15 = torch::empty({7, 3072}, input0.options());
    auto down15 = torch::empty({7, 1024}, input0.options());
    auto hidden16 = torch::empty({7, 1024}, input0.options());
    auto norm16 = torch::empty({7, 1024}, input0.options());
    auto qkv16 = torch::empty({7, 4096}, input0.options());
    auto attn16 = torch::empty({7, 2048}, input0.options());
    auto projected16 = torch::empty({7, 1024}, input0.options());
    auto residual16 = torch::empty({7, 1024}, input0.options());
    auto postnorm16 = torch::empty({7, 1024}, input0.options());
    auto gateup16 = torch::empty({7, 6144}, input0.options());
    auto activated16 = torch::empty({7, 3072}, input0.options());
    auto down16 = torch::empty({7, 1024}, input0.options());
    auto hidden17 = torch::empty({7, 1024}, input0.options());
    auto norm17 = torch::empty({7, 1024}, input0.options());
    auto qkv17 = torch::empty({7, 4096}, input0.options());
    auto attn17 = torch::empty({7, 2048}, input0.options());
    auto projected17 = torch::empty({7, 1024}, input0.options());
    auto residual17 = torch::empty({7, 1024}, input0.options());
    auto postnorm17 = torch::empty({7, 1024}, input0.options());
    auto gateup17 = torch::empty({7, 6144}, input0.options());
    auto activated17 = torch::empty({7, 3072}, input0.options());
    auto down17 = torch::empty({7, 1024}, input0.options());
    auto hidden18 = torch::empty({7, 1024}, input0.options());
    auto norm18 = torch::empty({7, 1024}, input0.options());
    auto qkv18 = torch::empty({7, 4096}, input0.options());
    auto attn18 = torch::empty({7, 2048}, input0.options());
    auto projected18 = torch::empty({7, 1024}, input0.options());
    auto residual18 = torch::empty({7, 1024}, input0.options());
    auto postnorm18 = torch::empty({7, 1024}, input0.options());
    auto gateup18 = torch::empty({7, 6144}, input0.options());
    auto activated18 = torch::empty({7, 3072}, input0.options());
    auto down18 = torch::empty({7, 1024}, input0.options());
    auto hidden19 = torch::empty({7, 1024}, input0.options());
    auto norm19 = torch::empty({7, 1024}, input0.options());
    auto qkv19 = torch::empty({7, 4096}, input0.options());
    auto attn19 = torch::empty({7, 2048}, input0.options());
    auto projected19 = torch::empty({7, 1024}, input0.options());
    auto residual19 = torch::empty({7, 1024}, input0.options());
    auto postnorm19 = torch::empty({7, 1024}, input0.options());
    auto gateup19 = torch::empty({7, 6144}, input0.options());
    auto activated19 = torch::empty({7, 3072}, input0.options());
    auto down19 = torch::empty({7, 1024}, input0.options());
    auto hidden20 = torch::empty({7, 1024}, input0.options());
    auto norm20 = torch::empty({7, 1024}, input0.options());
    auto qkv20 = torch::empty({7, 4096}, input0.options());
    auto attn20 = torch::empty({7, 2048}, input0.options());
    auto projected20 = torch::empty({7, 1024}, input0.options());
    auto residual20 = torch::empty({7, 1024}, input0.options());
    auto postnorm20 = torch::empty({7, 1024}, input0.options());
    auto gateup20 = torch::empty({7, 6144}, input0.options());
    auto activated20 = torch::empty({7, 3072}, input0.options());
    auto down20 = torch::empty({7, 1024}, input0.options());
    auto hidden21 = torch::empty({7, 1024}, input0.options());
    auto norm21 = torch::empty({7, 1024}, input0.options());
    auto qkv21 = torch::empty({7, 4096}, input0.options());
    auto attn21 = torch::empty({7, 2048}, input0.options());
    auto projected21 = torch::empty({7, 1024}, input0.options());
    auto residual21 = torch::empty({7, 1024}, input0.options());
    auto postnorm21 = torch::empty({7, 1024}, input0.options());
    auto gateup21 = torch::empty({7, 6144}, input0.options());
    auto activated21 = torch::empty({7, 3072}, input0.options());
    auto down21 = torch::empty({7, 1024}, input0.options());
    auto hidden22 = torch::empty({7, 1024}, input0.options());
    auto norm22 = torch::empty({7, 1024}, input0.options());
    auto qkv22 = torch::empty({7, 4096}, input0.options());
    auto attn22 = torch::empty({7, 2048}, input0.options());
    auto projected22 = torch::empty({7, 1024}, input0.options());
    auto residual22 = torch::empty({7, 1024}, input0.options());
    auto postnorm22 = torch::empty({7, 1024}, input0.options());
    auto gateup22 = torch::empty({7, 6144}, input0.options());
    auto activated22 = torch::empty({7, 3072}, input0.options());
    auto down22 = torch::empty({7, 1024}, input0.options());
    auto hidden23 = torch::empty({7, 1024}, input0.options());
    auto norm23 = torch::empty({7, 1024}, input0.options());
    auto qkv23 = torch::empty({7, 4096}, input0.options());
    auto attn23 = torch::empty({7, 2048}, input0.options());
    auto projected23 = torch::empty({7, 1024}, input0.options());
    auto residual23 = torch::empty({7, 1024}, input0.options());
    auto postnorm23 = torch::empty({7, 1024}, input0.options());
    auto gateup23 = torch::empty({7, 6144}, input0.options());
    auto activated23 = torch::empty({7, 3072}, input0.options());
    auto down23 = torch::empty({7, 1024}, input0.options());
    auto hidden24 = torch::empty({7, 1024}, input0.options());
    auto norm24 = torch::empty({7, 1024}, input0.options());
    auto qkv24 = torch::empty({7, 4096}, input0.options());
    auto attn24 = torch::empty({7, 2048}, input0.options());
    auto projected24 = torch::empty({7, 1024}, input0.options());
    auto residual24 = torch::empty({7, 1024}, input0.options());
    auto postnorm24 = torch::empty({7, 1024}, input0.options());
    auto gateup24 = torch::empty({7, 6144}, input0.options());
    auto activated24 = torch::empty({7, 3072}, input0.options());
    auto down24 = torch::empty({7, 1024}, input0.options());
    auto hidden25 = torch::empty({7, 1024}, input0.options());
    auto norm25 = torch::empty({7, 1024}, input0.options());
    auto qkv25 = torch::empty({7, 4096}, input0.options());
    auto attn25 = torch::empty({7, 2048}, input0.options());
    auto projected25 = torch::empty({7, 1024}, input0.options());
    auto residual25 = torch::empty({7, 1024}, input0.options());
    auto postnorm25 = torch::empty({7, 1024}, input0.options());
    auto gateup25 = torch::empty({7, 6144}, input0.options());
    auto activated25 = torch::empty({7, 3072}, input0.options());
    auto down25 = torch::empty({7, 1024}, input0.options());
    auto hidden26 = torch::empty({7, 1024}, input0.options());
    auto norm26 = torch::empty({7, 1024}, input0.options());
    auto qkv26 = torch::empty({7, 4096}, input0.options());
    auto attn26 = torch::empty({7, 2048}, input0.options());
    auto projected26 = torch::empty({7, 1024}, input0.options());
    auto residual26 = torch::empty({7, 1024}, input0.options());
    auto postnorm26 = torch::empty({7, 1024}, input0.options());
    auto gateup26 = torch::empty({7, 6144}, input0.options());
    auto activated26 = torch::empty({7, 3072}, input0.options());
    auto down26 = torch::empty({7, 1024}, input0.options());
    auto hidden27 = torch::empty({7, 1024}, input0.options());
    auto norm27 = torch::empty({7, 1024}, input0.options());
    auto qkv27 = torch::empty({7, 4096}, input0.options());
    auto attn27 = torch::empty({7, 2048}, input0.options());
    auto projected27 = torch::empty({7, 1024}, input0.options());
    auto residual27 = torch::empty({7, 1024}, input0.options());
    auto postnorm27 = torch::empty({7, 1024}, input0.options());
    auto gateup27 = torch::empty({7, 6144}, input0.options());
    auto activated27 = torch::empty({7, 3072}, input0.options());
    auto down27 = torch::empty({7, 1024}, input0.options());
    auto hidden28 = torch::empty({7, 1024}, input0.options());
    auto output_hidden = torch::empty({7, 1024}, input0.options());
    auto logits = torch::empty({7, 151936}, input0.options());
    auto output_tokens = torch::empty({1, 8}, input1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    embedding<<<1, 1, 0, stream>>>(input0.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), hidden0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>());
    rmsnorm<<<1, 1, 0, stream>>>(hidden0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), norm0.data_ptr<at::BFloat16>(), 155583488);
    linear<<<1, 32, 0, stream>>>(norm0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv0.data_ptr<at::BFloat16>(), 4096, 1024, 155585536, 1);
    attention<<<1, 1, 0, stream>>>(qkv0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn0.data_ptr<at::BFloat16>(), 161876992);
    linear<<<1, 32, 0, stream>>>(attn0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected0.data_ptr<at::BFloat16>(), 1024, 2048, 159779840, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden0.data_ptr<at::BFloat16>(), projected0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual0.data_ptr<at::BFloat16>(), postnorm0.data_ptr<at::BFloat16>(), 155584512);
    linear<<<1, 32, 0, stream>>>(postnorm0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup0.data_ptr<at::BFloat16>(), 6144, 1024, 161877248, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup0.data_ptr<at::BFloat16>(), activated0.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down0.data_ptr<at::BFloat16>(), 1024, 3072, 168168704, 2);
    add_norm<<<1, 1, 0, stream>>>(residual0.data_ptr<at::BFloat16>(), down0.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden1.data_ptr<at::BFloat16>(), norm1.data_ptr<at::BFloat16>(), 171314432);
    linear<<<1, 32, 0, stream>>>(norm1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv1.data_ptr<at::BFloat16>(), 4096, 1024, 171316480, 1);
    attention<<<1, 1, 0, stream>>>(qkv1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn1.data_ptr<at::BFloat16>(), 177607936);
    linear<<<1, 32, 0, stream>>>(attn1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected1.data_ptr<at::BFloat16>(), 1024, 2048, 175510784, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden1.data_ptr<at::BFloat16>(), projected1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual1.data_ptr<at::BFloat16>(), postnorm1.data_ptr<at::BFloat16>(), 171315456);
    linear<<<1, 32, 0, stream>>>(postnorm1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup1.data_ptr<at::BFloat16>(), 6144, 1024, 177608192, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup1.data_ptr<at::BFloat16>(), activated1.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down1.data_ptr<at::BFloat16>(), 1024, 3072, 183899648, 2);
    add_norm<<<1, 1, 0, stream>>>(residual1.data_ptr<at::BFloat16>(), down1.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden2.data_ptr<at::BFloat16>(), norm2.data_ptr<at::BFloat16>(), 187045376);
    linear<<<1, 32, 0, stream>>>(norm2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv2.data_ptr<at::BFloat16>(), 4096, 1024, 187047424, 1);
    attention<<<1, 1, 0, stream>>>(qkv2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn2.data_ptr<at::BFloat16>(), 193338880);
    linear<<<1, 32, 0, stream>>>(attn2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected2.data_ptr<at::BFloat16>(), 1024, 2048, 191241728, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden2.data_ptr<at::BFloat16>(), projected2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual2.data_ptr<at::BFloat16>(), postnorm2.data_ptr<at::BFloat16>(), 187046400);
    linear<<<1, 32, 0, stream>>>(postnorm2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup2.data_ptr<at::BFloat16>(), 6144, 1024, 193339136, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup2.data_ptr<at::BFloat16>(), activated2.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down2.data_ptr<at::BFloat16>(), 1024, 3072, 199630592, 2);
    add_norm<<<1, 1, 0, stream>>>(residual2.data_ptr<at::BFloat16>(), down2.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden3.data_ptr<at::BFloat16>(), norm3.data_ptr<at::BFloat16>(), 202776320);
    linear<<<1, 32, 0, stream>>>(norm3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv3.data_ptr<at::BFloat16>(), 4096, 1024, 202778368, 1);
    attention<<<1, 1, 0, stream>>>(qkv3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn3.data_ptr<at::BFloat16>(), 209069824);
    linear<<<1, 32, 0, stream>>>(attn3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected3.data_ptr<at::BFloat16>(), 1024, 2048, 206972672, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden3.data_ptr<at::BFloat16>(), projected3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual3.data_ptr<at::BFloat16>(), postnorm3.data_ptr<at::BFloat16>(), 202777344);
    linear<<<1, 32, 0, stream>>>(postnorm3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup3.data_ptr<at::BFloat16>(), 6144, 1024, 209070080, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup3.data_ptr<at::BFloat16>(), activated3.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down3.data_ptr<at::BFloat16>(), 1024, 3072, 215361536, 2);
    add_norm<<<1, 1, 0, stream>>>(residual3.data_ptr<at::BFloat16>(), down3.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden4.data_ptr<at::BFloat16>(), norm4.data_ptr<at::BFloat16>(), 218507264);
    linear<<<1, 32, 0, stream>>>(norm4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv4.data_ptr<at::BFloat16>(), 4096, 1024, 218509312, 1);
    attention<<<1, 1, 0, stream>>>(qkv4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn4.data_ptr<at::BFloat16>(), 224800768);
    linear<<<1, 32, 0, stream>>>(attn4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected4.data_ptr<at::BFloat16>(), 1024, 2048, 222703616, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden4.data_ptr<at::BFloat16>(), projected4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual4.data_ptr<at::BFloat16>(), postnorm4.data_ptr<at::BFloat16>(), 218508288);
    linear<<<1, 32, 0, stream>>>(postnorm4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup4.data_ptr<at::BFloat16>(), 6144, 1024, 224801024, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup4.data_ptr<at::BFloat16>(), activated4.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down4.data_ptr<at::BFloat16>(), 1024, 3072, 231092480, 2);
    add_norm<<<1, 1, 0, stream>>>(residual4.data_ptr<at::BFloat16>(), down4.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden5.data_ptr<at::BFloat16>(), norm5.data_ptr<at::BFloat16>(), 234238208);
    linear<<<1, 32, 0, stream>>>(norm5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv5.data_ptr<at::BFloat16>(), 4096, 1024, 234240256, 1);
    attention<<<1, 1, 0, stream>>>(qkv5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn5.data_ptr<at::BFloat16>(), 240531712);
    linear<<<1, 32, 0, stream>>>(attn5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected5.data_ptr<at::BFloat16>(), 1024, 2048, 238434560, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden5.data_ptr<at::BFloat16>(), projected5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual5.data_ptr<at::BFloat16>(), postnorm5.data_ptr<at::BFloat16>(), 234239232);
    linear<<<1, 32, 0, stream>>>(postnorm5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup5.data_ptr<at::BFloat16>(), 6144, 1024, 240531968, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup5.data_ptr<at::BFloat16>(), activated5.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down5.data_ptr<at::BFloat16>(), 1024, 3072, 246823424, 2);
    add_norm<<<1, 1, 0, stream>>>(residual5.data_ptr<at::BFloat16>(), down5.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden6.data_ptr<at::BFloat16>(), norm6.data_ptr<at::BFloat16>(), 249969152);
    linear<<<1, 32, 0, stream>>>(norm6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv6.data_ptr<at::BFloat16>(), 4096, 1024, 249971200, 1);
    attention<<<1, 1, 0, stream>>>(qkv6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn6.data_ptr<at::BFloat16>(), 256262656);
    linear<<<1, 32, 0, stream>>>(attn6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected6.data_ptr<at::BFloat16>(), 1024, 2048, 254165504, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden6.data_ptr<at::BFloat16>(), projected6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual6.data_ptr<at::BFloat16>(), postnorm6.data_ptr<at::BFloat16>(), 249970176);
    linear<<<1, 32, 0, stream>>>(postnorm6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup6.data_ptr<at::BFloat16>(), 6144, 1024, 256262912, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup6.data_ptr<at::BFloat16>(), activated6.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down6.data_ptr<at::BFloat16>(), 1024, 3072, 262554368, 2);
    add_norm<<<1, 1, 0, stream>>>(residual6.data_ptr<at::BFloat16>(), down6.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden7.data_ptr<at::BFloat16>(), norm7.data_ptr<at::BFloat16>(), 265700096);
    linear<<<1, 32, 0, stream>>>(norm7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv7.data_ptr<at::BFloat16>(), 4096, 1024, 265702144, 1);
    attention<<<1, 1, 0, stream>>>(qkv7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn7.data_ptr<at::BFloat16>(), 271993600);
    linear<<<1, 32, 0, stream>>>(attn7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected7.data_ptr<at::BFloat16>(), 1024, 2048, 269896448, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden7.data_ptr<at::BFloat16>(), projected7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual7.data_ptr<at::BFloat16>(), postnorm7.data_ptr<at::BFloat16>(), 265701120);
    linear<<<1, 32, 0, stream>>>(postnorm7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup7.data_ptr<at::BFloat16>(), 6144, 1024, 271993856, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup7.data_ptr<at::BFloat16>(), activated7.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down7.data_ptr<at::BFloat16>(), 1024, 3072, 278285312, 2);
    add_norm<<<1, 1, 0, stream>>>(residual7.data_ptr<at::BFloat16>(), down7.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden8.data_ptr<at::BFloat16>(), norm8.data_ptr<at::BFloat16>(), 281431040);
    linear<<<1, 32, 0, stream>>>(norm8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv8.data_ptr<at::BFloat16>(), 4096, 1024, 281433088, 1);
    attention<<<1, 1, 0, stream>>>(qkv8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn8.data_ptr<at::BFloat16>(), 287724544);
    linear<<<1, 32, 0, stream>>>(attn8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected8.data_ptr<at::BFloat16>(), 1024, 2048, 285627392, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden8.data_ptr<at::BFloat16>(), projected8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual8.data_ptr<at::BFloat16>(), postnorm8.data_ptr<at::BFloat16>(), 281432064);
    linear<<<1, 32, 0, stream>>>(postnorm8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup8.data_ptr<at::BFloat16>(), 6144, 1024, 287724800, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup8.data_ptr<at::BFloat16>(), activated8.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down8.data_ptr<at::BFloat16>(), 1024, 3072, 294016256, 2);
    add_norm<<<1, 1, 0, stream>>>(residual8.data_ptr<at::BFloat16>(), down8.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden9.data_ptr<at::BFloat16>(), norm9.data_ptr<at::BFloat16>(), 297161984);
    linear<<<1, 32, 0, stream>>>(norm9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv9.data_ptr<at::BFloat16>(), 4096, 1024, 297164032, 1);
    attention<<<1, 1, 0, stream>>>(qkv9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn9.data_ptr<at::BFloat16>(), 303455488);
    linear<<<1, 32, 0, stream>>>(attn9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected9.data_ptr<at::BFloat16>(), 1024, 2048, 301358336, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden9.data_ptr<at::BFloat16>(), projected9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual9.data_ptr<at::BFloat16>(), postnorm9.data_ptr<at::BFloat16>(), 297163008);
    linear<<<1, 32, 0, stream>>>(postnorm9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup9.data_ptr<at::BFloat16>(), 6144, 1024, 303455744, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup9.data_ptr<at::BFloat16>(), activated9.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down9.data_ptr<at::BFloat16>(), 1024, 3072, 309747200, 2);
    add_norm<<<1, 1, 0, stream>>>(residual9.data_ptr<at::BFloat16>(), down9.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden10.data_ptr<at::BFloat16>(), norm10.data_ptr<at::BFloat16>(), 312892928);
    linear<<<1, 32, 0, stream>>>(norm10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv10.data_ptr<at::BFloat16>(), 4096, 1024, 312894976, 1);
    attention<<<1, 1, 0, stream>>>(qkv10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn10.data_ptr<at::BFloat16>(), 319186432);
    linear<<<1, 32, 0, stream>>>(attn10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected10.data_ptr<at::BFloat16>(), 1024, 2048, 317089280, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden10.data_ptr<at::BFloat16>(), projected10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual10.data_ptr<at::BFloat16>(), postnorm10.data_ptr<at::BFloat16>(), 312893952);
    linear<<<1, 32, 0, stream>>>(postnorm10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup10.data_ptr<at::BFloat16>(), 6144, 1024, 319186688, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup10.data_ptr<at::BFloat16>(), activated10.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down10.data_ptr<at::BFloat16>(), 1024, 3072, 325478144, 2);
    add_norm<<<1, 1, 0, stream>>>(residual10.data_ptr<at::BFloat16>(), down10.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden11.data_ptr<at::BFloat16>(), norm11.data_ptr<at::BFloat16>(), 328623872);
    linear<<<1, 32, 0, stream>>>(norm11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv11.data_ptr<at::BFloat16>(), 4096, 1024, 328625920, 1);
    attention<<<1, 1, 0, stream>>>(qkv11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn11.data_ptr<at::BFloat16>(), 334917376);
    linear<<<1, 32, 0, stream>>>(attn11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected11.data_ptr<at::BFloat16>(), 1024, 2048, 332820224, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden11.data_ptr<at::BFloat16>(), projected11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual11.data_ptr<at::BFloat16>(), postnorm11.data_ptr<at::BFloat16>(), 328624896);
    linear<<<1, 32, 0, stream>>>(postnorm11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup11.data_ptr<at::BFloat16>(), 6144, 1024, 334917632, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup11.data_ptr<at::BFloat16>(), activated11.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down11.data_ptr<at::BFloat16>(), 1024, 3072, 341209088, 2);
    add_norm<<<1, 1, 0, stream>>>(residual11.data_ptr<at::BFloat16>(), down11.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden12.data_ptr<at::BFloat16>(), norm12.data_ptr<at::BFloat16>(), 344354816);
    linear<<<1, 32, 0, stream>>>(norm12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv12.data_ptr<at::BFloat16>(), 4096, 1024, 344356864, 1);
    attention<<<1, 1, 0, stream>>>(qkv12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn12.data_ptr<at::BFloat16>(), 350648320);
    linear<<<1, 32, 0, stream>>>(attn12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected12.data_ptr<at::BFloat16>(), 1024, 2048, 348551168, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden12.data_ptr<at::BFloat16>(), projected12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual12.data_ptr<at::BFloat16>(), postnorm12.data_ptr<at::BFloat16>(), 344355840);
    linear<<<1, 32, 0, stream>>>(postnorm12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup12.data_ptr<at::BFloat16>(), 6144, 1024, 350648576, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup12.data_ptr<at::BFloat16>(), activated12.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down12.data_ptr<at::BFloat16>(), 1024, 3072, 356940032, 2);
    add_norm<<<1, 1, 0, stream>>>(residual12.data_ptr<at::BFloat16>(), down12.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden13.data_ptr<at::BFloat16>(), norm13.data_ptr<at::BFloat16>(), 360085760);
    linear<<<1, 32, 0, stream>>>(norm13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv13.data_ptr<at::BFloat16>(), 4096, 1024, 360087808, 1);
    attention<<<1, 1, 0, stream>>>(qkv13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn13.data_ptr<at::BFloat16>(), 366379264);
    linear<<<1, 32, 0, stream>>>(attn13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected13.data_ptr<at::BFloat16>(), 1024, 2048, 364282112, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden13.data_ptr<at::BFloat16>(), projected13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual13.data_ptr<at::BFloat16>(), postnorm13.data_ptr<at::BFloat16>(), 360086784);
    linear<<<1, 32, 0, stream>>>(postnorm13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup13.data_ptr<at::BFloat16>(), 6144, 1024, 366379520, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup13.data_ptr<at::BFloat16>(), activated13.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down13.data_ptr<at::BFloat16>(), 1024, 3072, 372670976, 2);
    add_norm<<<1, 1, 0, stream>>>(residual13.data_ptr<at::BFloat16>(), down13.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden14.data_ptr<at::BFloat16>(), norm14.data_ptr<at::BFloat16>(), 375816704);
    linear<<<1, 32, 0, stream>>>(norm14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv14.data_ptr<at::BFloat16>(), 4096, 1024, 375818752, 1);
    attention<<<1, 1, 0, stream>>>(qkv14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn14.data_ptr<at::BFloat16>(), 382110208);
    linear<<<1, 32, 0, stream>>>(attn14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected14.data_ptr<at::BFloat16>(), 1024, 2048, 380013056, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden14.data_ptr<at::BFloat16>(), projected14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual14.data_ptr<at::BFloat16>(), postnorm14.data_ptr<at::BFloat16>(), 375817728);
    linear<<<1, 32, 0, stream>>>(postnorm14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup14.data_ptr<at::BFloat16>(), 6144, 1024, 382110464, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup14.data_ptr<at::BFloat16>(), activated14.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down14.data_ptr<at::BFloat16>(), 1024, 3072, 388401920, 2);
    add_norm<<<1, 1, 0, stream>>>(residual14.data_ptr<at::BFloat16>(), down14.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden15.data_ptr<at::BFloat16>(), norm15.data_ptr<at::BFloat16>(), 391547648);
    linear<<<1, 32, 0, stream>>>(norm15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv15.data_ptr<at::BFloat16>(), 4096, 1024, 391549696, 1);
    attention<<<1, 1, 0, stream>>>(qkv15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn15.data_ptr<at::BFloat16>(), 397841152);
    linear<<<1, 32, 0, stream>>>(attn15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected15.data_ptr<at::BFloat16>(), 1024, 2048, 395744000, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden15.data_ptr<at::BFloat16>(), projected15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual15.data_ptr<at::BFloat16>(), postnorm15.data_ptr<at::BFloat16>(), 391548672);
    linear<<<1, 32, 0, stream>>>(postnorm15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup15.data_ptr<at::BFloat16>(), 6144, 1024, 397841408, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup15.data_ptr<at::BFloat16>(), activated15.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down15.data_ptr<at::BFloat16>(), 1024, 3072, 404132864, 2);
    add_norm<<<1, 1, 0, stream>>>(residual15.data_ptr<at::BFloat16>(), down15.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden16.data_ptr<at::BFloat16>(), norm16.data_ptr<at::BFloat16>(), 407278592);
    linear<<<1, 32, 0, stream>>>(norm16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv16.data_ptr<at::BFloat16>(), 4096, 1024, 407280640, 1);
    attention<<<1, 1, 0, stream>>>(qkv16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn16.data_ptr<at::BFloat16>(), 413572096);
    linear<<<1, 32, 0, stream>>>(attn16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected16.data_ptr<at::BFloat16>(), 1024, 2048, 411474944, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden16.data_ptr<at::BFloat16>(), projected16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual16.data_ptr<at::BFloat16>(), postnorm16.data_ptr<at::BFloat16>(), 407279616);
    linear<<<1, 32, 0, stream>>>(postnorm16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup16.data_ptr<at::BFloat16>(), 6144, 1024, 413572352, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup16.data_ptr<at::BFloat16>(), activated16.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down16.data_ptr<at::BFloat16>(), 1024, 3072, 419863808, 2);
    add_norm<<<1, 1, 0, stream>>>(residual16.data_ptr<at::BFloat16>(), down16.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden17.data_ptr<at::BFloat16>(), norm17.data_ptr<at::BFloat16>(), 423009536);
    linear<<<1, 32, 0, stream>>>(norm17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv17.data_ptr<at::BFloat16>(), 4096, 1024, 423011584, 1);
    attention<<<1, 1, 0, stream>>>(qkv17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn17.data_ptr<at::BFloat16>(), 429303040);
    linear<<<1, 32, 0, stream>>>(attn17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected17.data_ptr<at::BFloat16>(), 1024, 2048, 427205888, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden17.data_ptr<at::BFloat16>(), projected17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual17.data_ptr<at::BFloat16>(), postnorm17.data_ptr<at::BFloat16>(), 423010560);
    linear<<<1, 32, 0, stream>>>(postnorm17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup17.data_ptr<at::BFloat16>(), 6144, 1024, 429303296, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup17.data_ptr<at::BFloat16>(), activated17.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down17.data_ptr<at::BFloat16>(), 1024, 3072, 435594752, 2);
    add_norm<<<1, 1, 0, stream>>>(residual17.data_ptr<at::BFloat16>(), down17.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden18.data_ptr<at::BFloat16>(), norm18.data_ptr<at::BFloat16>(), 438740480);
    linear<<<1, 32, 0, stream>>>(norm18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv18.data_ptr<at::BFloat16>(), 4096, 1024, 438742528, 1);
    attention<<<1, 1, 0, stream>>>(qkv18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn18.data_ptr<at::BFloat16>(), 445033984);
    linear<<<1, 32, 0, stream>>>(attn18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected18.data_ptr<at::BFloat16>(), 1024, 2048, 442936832, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden18.data_ptr<at::BFloat16>(), projected18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual18.data_ptr<at::BFloat16>(), postnorm18.data_ptr<at::BFloat16>(), 438741504);
    linear<<<1, 32, 0, stream>>>(postnorm18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup18.data_ptr<at::BFloat16>(), 6144, 1024, 445034240, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup18.data_ptr<at::BFloat16>(), activated18.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down18.data_ptr<at::BFloat16>(), 1024, 3072, 451325696, 2);
    add_norm<<<1, 1, 0, stream>>>(residual18.data_ptr<at::BFloat16>(), down18.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden19.data_ptr<at::BFloat16>(), norm19.data_ptr<at::BFloat16>(), 454471424);
    linear<<<1, 32, 0, stream>>>(norm19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv19.data_ptr<at::BFloat16>(), 4096, 1024, 454473472, 1);
    attention<<<1, 1, 0, stream>>>(qkv19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn19.data_ptr<at::BFloat16>(), 460764928);
    linear<<<1, 32, 0, stream>>>(attn19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected19.data_ptr<at::BFloat16>(), 1024, 2048, 458667776, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden19.data_ptr<at::BFloat16>(), projected19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual19.data_ptr<at::BFloat16>(), postnorm19.data_ptr<at::BFloat16>(), 454472448);
    linear<<<1, 32, 0, stream>>>(postnorm19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup19.data_ptr<at::BFloat16>(), 6144, 1024, 460765184, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup19.data_ptr<at::BFloat16>(), activated19.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down19.data_ptr<at::BFloat16>(), 1024, 3072, 467056640, 2);
    add_norm<<<1, 1, 0, stream>>>(residual19.data_ptr<at::BFloat16>(), down19.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden20.data_ptr<at::BFloat16>(), norm20.data_ptr<at::BFloat16>(), 470202368);
    linear<<<1, 32, 0, stream>>>(norm20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv20.data_ptr<at::BFloat16>(), 4096, 1024, 470204416, 1);
    attention<<<1, 1, 0, stream>>>(qkv20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn20.data_ptr<at::BFloat16>(), 476495872);
    linear<<<1, 32, 0, stream>>>(attn20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected20.data_ptr<at::BFloat16>(), 1024, 2048, 474398720, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden20.data_ptr<at::BFloat16>(), projected20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual20.data_ptr<at::BFloat16>(), postnorm20.data_ptr<at::BFloat16>(), 470203392);
    linear<<<1, 32, 0, stream>>>(postnorm20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup20.data_ptr<at::BFloat16>(), 6144, 1024, 476496128, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup20.data_ptr<at::BFloat16>(), activated20.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down20.data_ptr<at::BFloat16>(), 1024, 3072, 482787584, 2);
    add_norm<<<1, 1, 0, stream>>>(residual20.data_ptr<at::BFloat16>(), down20.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden21.data_ptr<at::BFloat16>(), norm21.data_ptr<at::BFloat16>(), 485933312);
    linear<<<1, 32, 0, stream>>>(norm21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv21.data_ptr<at::BFloat16>(), 4096, 1024, 485935360, 1);
    attention<<<1, 1, 0, stream>>>(qkv21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn21.data_ptr<at::BFloat16>(), 492226816);
    linear<<<1, 32, 0, stream>>>(attn21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected21.data_ptr<at::BFloat16>(), 1024, 2048, 490129664, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden21.data_ptr<at::BFloat16>(), projected21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual21.data_ptr<at::BFloat16>(), postnorm21.data_ptr<at::BFloat16>(), 485934336);
    linear<<<1, 32, 0, stream>>>(postnorm21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup21.data_ptr<at::BFloat16>(), 6144, 1024, 492227072, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup21.data_ptr<at::BFloat16>(), activated21.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down21.data_ptr<at::BFloat16>(), 1024, 3072, 498518528, 2);
    add_norm<<<1, 1, 0, stream>>>(residual21.data_ptr<at::BFloat16>(), down21.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden22.data_ptr<at::BFloat16>(), norm22.data_ptr<at::BFloat16>(), 501664256);
    linear<<<1, 32, 0, stream>>>(norm22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv22.data_ptr<at::BFloat16>(), 4096, 1024, 501666304, 1);
    attention<<<1, 1, 0, stream>>>(qkv22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn22.data_ptr<at::BFloat16>(), 507957760);
    linear<<<1, 32, 0, stream>>>(attn22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected22.data_ptr<at::BFloat16>(), 1024, 2048, 505860608, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden22.data_ptr<at::BFloat16>(), projected22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual22.data_ptr<at::BFloat16>(), postnorm22.data_ptr<at::BFloat16>(), 501665280);
    linear<<<1, 32, 0, stream>>>(postnorm22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup22.data_ptr<at::BFloat16>(), 6144, 1024, 507958016, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup22.data_ptr<at::BFloat16>(), activated22.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down22.data_ptr<at::BFloat16>(), 1024, 3072, 514249472, 2);
    add_norm<<<1, 1, 0, stream>>>(residual22.data_ptr<at::BFloat16>(), down22.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden23.data_ptr<at::BFloat16>(), norm23.data_ptr<at::BFloat16>(), 517395200);
    linear<<<1, 32, 0, stream>>>(norm23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv23.data_ptr<at::BFloat16>(), 4096, 1024, 517397248, 1);
    attention<<<1, 1, 0, stream>>>(qkv23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn23.data_ptr<at::BFloat16>(), 523688704);
    linear<<<1, 32, 0, stream>>>(attn23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected23.data_ptr<at::BFloat16>(), 1024, 2048, 521591552, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden23.data_ptr<at::BFloat16>(), projected23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual23.data_ptr<at::BFloat16>(), postnorm23.data_ptr<at::BFloat16>(), 517396224);
    linear<<<1, 32, 0, stream>>>(postnorm23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup23.data_ptr<at::BFloat16>(), 6144, 1024, 523688960, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup23.data_ptr<at::BFloat16>(), activated23.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down23.data_ptr<at::BFloat16>(), 1024, 3072, 529980416, 2);
    add_norm<<<1, 1, 0, stream>>>(residual23.data_ptr<at::BFloat16>(), down23.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden24.data_ptr<at::BFloat16>(), norm24.data_ptr<at::BFloat16>(), 533126144);
    linear<<<1, 32, 0, stream>>>(norm24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv24.data_ptr<at::BFloat16>(), 4096, 1024, 533128192, 1);
    attention<<<1, 1, 0, stream>>>(qkv24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn24.data_ptr<at::BFloat16>(), 539419648);
    linear<<<1, 32, 0, stream>>>(attn24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected24.data_ptr<at::BFloat16>(), 1024, 2048, 537322496, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden24.data_ptr<at::BFloat16>(), projected24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual24.data_ptr<at::BFloat16>(), postnorm24.data_ptr<at::BFloat16>(), 533127168);
    linear<<<1, 32, 0, stream>>>(postnorm24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup24.data_ptr<at::BFloat16>(), 6144, 1024, 539419904, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup24.data_ptr<at::BFloat16>(), activated24.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down24.data_ptr<at::BFloat16>(), 1024, 3072, 545711360, 2);
    add_norm<<<1, 1, 0, stream>>>(residual24.data_ptr<at::BFloat16>(), down24.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden25.data_ptr<at::BFloat16>(), norm25.data_ptr<at::BFloat16>(), 548857088);
    linear<<<1, 32, 0, stream>>>(norm25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv25.data_ptr<at::BFloat16>(), 4096, 1024, 548859136, 1);
    attention<<<1, 1, 0, stream>>>(qkv25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn25.data_ptr<at::BFloat16>(), 555150592);
    linear<<<1, 32, 0, stream>>>(attn25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected25.data_ptr<at::BFloat16>(), 1024, 2048, 553053440, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden25.data_ptr<at::BFloat16>(), projected25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual25.data_ptr<at::BFloat16>(), postnorm25.data_ptr<at::BFloat16>(), 548858112);
    linear<<<1, 32, 0, stream>>>(postnorm25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup25.data_ptr<at::BFloat16>(), 6144, 1024, 555150848, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup25.data_ptr<at::BFloat16>(), activated25.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down25.data_ptr<at::BFloat16>(), 1024, 3072, 561442304, 2);
    add_norm<<<1, 1, 0, stream>>>(residual25.data_ptr<at::BFloat16>(), down25.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden26.data_ptr<at::BFloat16>(), norm26.data_ptr<at::BFloat16>(), 564588032);
    linear<<<1, 32, 0, stream>>>(norm26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv26.data_ptr<at::BFloat16>(), 4096, 1024, 564590080, 1);
    attention<<<1, 1, 0, stream>>>(qkv26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn26.data_ptr<at::BFloat16>(), 570881536);
    linear<<<1, 32, 0, stream>>>(attn26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected26.data_ptr<at::BFloat16>(), 1024, 2048, 568784384, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden26.data_ptr<at::BFloat16>(), projected26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual26.data_ptr<at::BFloat16>(), postnorm26.data_ptr<at::BFloat16>(), 564589056);
    linear<<<1, 32, 0, stream>>>(postnorm26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup26.data_ptr<at::BFloat16>(), 6144, 1024, 570881792, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup26.data_ptr<at::BFloat16>(), activated26.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down26.data_ptr<at::BFloat16>(), 1024, 3072, 577173248, 2);
    add_norm<<<1, 1, 0, stream>>>(residual26.data_ptr<at::BFloat16>(), down26.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), hidden27.data_ptr<at::BFloat16>(), norm27.data_ptr<at::BFloat16>(), 580318976);
    linear<<<1, 32, 0, stream>>>(norm27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), qkv27.data_ptr<at::BFloat16>(), 4096, 1024, 580321024, 1);
    attention<<<1, 1, 0, stream>>>(qkv27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), rotary.data_ptr<at::BFloat16>(), attn27.data_ptr<at::BFloat16>(), 586612480);
    linear<<<1, 32, 0, stream>>>(attn27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), projected27.data_ptr<at::BFloat16>(), 1024, 2048, 584515328, 1);
    add_norm<<<1, 1, 0, stream>>>(hidden27.data_ptr<at::BFloat16>(), projected27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), residual27.data_ptr<at::BFloat16>(), postnorm27.data_ptr<at::BFloat16>(), 580320000);
    linear<<<1, 32, 0, stream>>>(postnorm27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), gateup27.data_ptr<at::BFloat16>(), 6144, 1024, 586612736, 1);
    silu_product<<<1, 1, 0, stream>>>(gateup27.data_ptr<at::BFloat16>(), activated27.data_ptr<at::BFloat16>());
    linear<<<1, 32, 0, stream>>>(activated27.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), down27.data_ptr<at::BFloat16>(), 1024, 3072, 592904192, 2);
    residual_add<<<1, 1, 0, stream>>>(residual27.data_ptr<at::BFloat16>(), down27.data_ptr<at::BFloat16>(), hidden28.data_ptr<at::BFloat16>());
    rmsnorm<<<1, 1, 0, stream>>>(hidden28.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), output_hidden.data_ptr<at::BFloat16>(), 155582464);
    linear<<<1, 32, 0, stream>>>(output_hidden.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(), logits.data_ptr<at::BFloat16>(), 151936, 1024, 0, 1);
    select_token<<<1, 1, 0, stream>>>(logits.data_ptr<at::BFloat16>(), input1.data_ptr<int64_t>(), output_tokens.data_ptr<int64_t>());
    return {output_tokens, output_hidden};
}
