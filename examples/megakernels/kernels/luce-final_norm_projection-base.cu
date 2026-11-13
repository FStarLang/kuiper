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

void check_tensor(const torch::Tensor& value, const torch::Tensor& owner,
                  at::ScalarType dtype, int64_t count, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == dtype
                && value.numel() == count, name, ": invalid device, layout, dtype or size");
    TORCH_CHECK(value.device() == owner.device(), name, ": tensors must share a CUDA device");
}

__global__ void projection_kernel(const at::BFloat16* weight, const at::BFloat16* input,
                                 const at::BFloat16* previous_hidden, at::BFloat16* scores,
                                 at::BFloat16* hidden, int position, int prompt_length) {
    __shared__ at::BFloat16 h[16000];
    int lane = threadIdx.x, group = lane / 4, pair = lane % 4;
    if (lane == 0) {
        int previous_count = position >= prompt_length ? position - prompt_length + 1 : 0;
        for (int i = 0; i < previous_count * 1024; ++i) hidden[i] = previous_hidden[i];
        for (int i = 0; i < 1024; ++i) h[i] = input[i];
        if (position >= prompt_length - 1)
            for (int i = 0; i < 1024; ++i)
                hidden[(position - prompt_length + 1) * 1024 + i] = h[i];
        apply_rms(weight, h, 0, 1024, 1024, 1, 1024, 254279680);
    }
    __syncwarp();
    bool final = true;
    int rows = 248320, columns = 1024, input_offset = 1024, output_offset = 12500;
    int weight_offset = 0, matrix = 0, splits = 1, layer = 0, head = 0;
    const at::BFloat16* cache = nullptr;
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
            int out = output_offset + (final ? 0 : row);
            if (row + group < rows) h[out + group] = at::BFloat16(total0);
            if (row + group + 8 < rows) h[out + group + 8] = at::BFloat16(total2);
        }
        if (final) {
            __syncwarp();
            if (lane == 0)
                for (int i = 0; i < 16; ++i)
                    scores[row + i] = h[12500 + i];
            __syncwarp();
        }
    }
}

std::vector<torch::Tensor> kplay_run(torch::Tensor input0, torch::Tensor input1,
                                    torch::Tensor input2, int64_t input3, int64_t input4) {
    TORCH_CHECK(input3 >= 0 && input3 < 2048 && input4 > 0 && input4 <= 2048,
                "invalid position or prompt length");
    int64_t previous_count = input3 >= input4 ? input3 - input4 + 1 : 0;
    int64_t count = input3 >= input4 - 1 ? input3 - input4 + 2 : 0;
    check_tensor(input0, input0, at::kBFloat16, 254280704, "embedding and final-norm weights");
    check_tensor(input1, input0, at::kBFloat16, 1024, "hidden before final norm");
    check_tensor(input2, input0, at::kBFloat16, previous_count * 1024, "emitted hidden prefix");
    c10::cuda::CUDAGuard guard(input0.device());
    auto scores = torch::empty({248320}, input0.options());
    auto hidden = torch::empty({count, 1024}, input0.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    projection_kernel<<<1, 32, 0, stream>>>(input0.data_ptr<at::BFloat16>(),
        input1.data_ptr<at::BFloat16>(), input2.data_ptr<at::BFloat16>(),
        scores.data_ptr<at::BFloat16>(), hidden.data_ptr<at::BFloat16>(), input3, input4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, hidden};
}
