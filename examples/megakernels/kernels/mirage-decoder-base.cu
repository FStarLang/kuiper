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

void check_bf16(const torch::Tensor& value, const torch::Tensor& weights) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() &&
                value.scalar_type() == at::kBFloat16 &&
                value.device() == weights.device(),
                "Expected contiguous BF16 tensors on the same CUDA device");
}

void check_state(const torch::Tensor& value, const torch::Tensor& weights) {
    check_bf16(value, weights);
    TORCH_CHECK(value.dim() == 2 && value.size(0) == 7 && value.size(1) == 1024,
                "Expected a [7, 1024] residual state");
}

void check_weight_span(const torch::Tensor& weights, int64_t offset, int64_t count) {
    TORCH_CHECK(offset >= 0 && offset <= weights.numel() - count,
                "Weight offset and extent exceed the supplied packed weights");
}

}  // namespace

std::vector<torch::Tensor> kplay_run(
    torch::Tensor input0, torch::Tensor input1, torch::Tensor input2,
    torch::Tensor input3, int64_t input4, int64_t input5, int64_t input6,
    int64_t input7, int64_t input8, int64_t input9, int64_t input10, bool input11) {
    check_bf16(input0, input0);
    TORCH_CHECK(input0.dim() == 1 && input0.numel() <= std::numeric_limits<int>::max(),
                "Packed layer weights must be a flat int-addressable tensor");
    check_state(input1, input0);
    check_state(input2, input0);
    check_bf16(input3, input0);
    TORCH_CHECK(input3.dim() == 3 && input3.size(0) == 2 &&
                input3.size(1) == 7 && input3.size(2) == 128,
                "Expected the full [2, 7, 128] rotary state");
    check_weight_span(input0, input4, 1024);
    check_weight_span(input0, input5, 4096 * 1024);
    check_weight_span(input0, input6, 256);
    check_weight_span(input0, input7, 1024 * 2048);
    check_weight_span(input0, input8, 1024);
    check_weight_span(input0, input9, 6144 * 1024);
    check_weight_span(input0, input10, 1024 * 3072);
    const c10::cuda::CUDAGuard guard(input0.device());
    auto hidden = input11 ? torch::empty({7, 1024}, input0.options()) : input1;
    auto normalized = torch::empty({7, 1024}, input0.options());
    auto qkv = torch::empty({7, 4096}, input0.options());
    auto attn = torch::empty({7, 2048}, input0.options());
    auto projected = torch::empty({7, 1024}, input0.options());
    auto residual = torch::empty({7, 1024}, input0.options());
    auto postnorm = torch::empty({7, 1024}, input0.options());
    auto gateup = torch::empty({7, 6144}, input0.options());
    auto activated = torch::empty({7, 3072}, input0.options());
    auto down = torch::empty({7, 1024}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    // Keep the accepted cross-layer residual-add/RMSNorm fusion intact.
    if (input11) {
        add_norm<<<1, 1, 0, stream>>>(
            input1.data_ptr<at::BFloat16>(), input2.data_ptr<at::BFloat16>(),
            input0.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(),
            normalized.data_ptr<at::BFloat16>(), static_cast<int>(input4));
    } else {
        rmsnorm<<<1, 1, 0, stream>>>(
            hidden.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
            normalized.data_ptr<at::BFloat16>(), static_cast<int>(input4));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    linear<<<1, 32, 0, stream>>>(
        normalized.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        qkv.data_ptr<at::BFloat16>(), 4096, 1024, static_cast<int>(input5), 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attention<<<1, 1, 0, stream>>>(
        qkv.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        input3.data_ptr<at::BFloat16>(), attn.data_ptr<at::BFloat16>(),
        static_cast<int>(input6));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    linear<<<1, 32, 0, stream>>>(
        attn.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        projected.data_ptr<at::BFloat16>(), 1024, 2048, static_cast<int>(input7), 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    add_norm<<<1, 1, 0, stream>>>(
        hidden.data_ptr<at::BFloat16>(), projected.data_ptr<at::BFloat16>(),
        input0.data_ptr<at::BFloat16>(), residual.data_ptr<at::BFloat16>(),
        postnorm.data_ptr<at::BFloat16>(), static_cast<int>(input8));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    linear<<<1, 32, 0, stream>>>(
        postnorm.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        gateup.data_ptr<at::BFloat16>(), 6144, 1024, static_cast<int>(input9), 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    silu_product<<<1, 1, 0, stream>>>(
        gateup.data_ptr<at::BFloat16>(), activated.data_ptr<at::BFloat16>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    linear<<<1, 32, 0, stream>>>(
        activated.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        down.data_ptr<at::BFloat16>(), 1024, 3072, static_cast<int>(input10), 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual, down};
}
