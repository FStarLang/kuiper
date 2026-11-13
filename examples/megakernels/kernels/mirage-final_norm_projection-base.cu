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

__global__ void residual_add(const at::BFloat16* residual, const at::BFloat16* delta,
                            at::BFloat16* output) {
    for (int i = 0; i < 7 * 1024; ++i)
        output[i] = at::BFloat16(__fadd_rn(float(residual[i]), float(delta[i])));
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
    int64_t input3, int64_t input4) {
    check_bf16(input0, input0);
    TORCH_CHECK(input0.dim() == 1 && input0.numel() <= std::numeric_limits<int>::max(),
                "Packed final weights must be a flat int-addressable tensor");
    check_state(input1, input0);
    check_state(input2, input0);
    check_weight_span(input0, input3, 1024);
    check_weight_span(input0, input4, 151936 * 1024);
    const c10::cuda::CUDAGuard guard(input0.device());
    auto hidden = torch::empty({7, 1024}, input0.options());
    auto output_hidden = torch::empty({7, 1024}, input0.options());
    auto logits = torch::empty({7, 151936}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    residual_add<<<1, 1, 0, stream>>>(
        input1.data_ptr<at::BFloat16>(), input2.data_ptr<at::BFloat16>(),
        hidden.data_ptr<at::BFloat16>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    rmsnorm<<<1, 1, 0, stream>>>(
        hidden.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        output_hidden.data_ptr<at::BFloat16>(), static_cast<int>(input3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    linear<<<1, 32, 0, stream>>>(
        output_hidden.data_ptr<at::BFloat16>(), input0.data_ptr<at::BFloat16>(),
        logits.data_ptr<at::BFloat16>(), 151936, 1024, static_cast<int>(input4), 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_hidden, logits};
}
