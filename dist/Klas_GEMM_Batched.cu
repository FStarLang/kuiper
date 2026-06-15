
#include "Klas_GEMM_Batched.h"

__global__
/**
  hoisted when extracting batched_gemm_f32
*/
static void
__hoisted_batched_gemm_f32_0(float alpha,
                             float beta,
                             uint32_t rows,
                             uint32_t shared,
                             uint32_t cols,
                             float *a, float *b, float *c, uint32_t i)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % cols;
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < shared; k++) {
            uint32_t vk = k;
            sum +=
                a[i * rows * shared + trow * shared +
                  vk] * b[i * shared * cols + vk * cols + tcol];
        }
        float s = sum;
        c[i * rows * cols + trow * cols + tcol] =
            beta * c[i * rows * cols + trow * cols + tcol] + alpha * s;
    }
}

void
Klas_GEMM_Batched_batched_gemm_f32(float alpha,
                                   float beta,
                                   uint32_t batch,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols, float *a, float *b, float *c)
{
    uint32_t idx = 0U;
    for (; idx < batch; idx++) {
        uint32_t i = idx;
        KPR_KCALL(__hoisted_batched_gemm_f32_0,
                  rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
                  1024U, 0U, alpha, beta, rows, shared, cols, a, b, c, i);
        MUST(cudaDeviceSynchronize());
    }
}

__global__
/**
  hoisted when extracting batched_matmul_f32
*/
static void
__hoisted_batched_matmul_f32_0(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               float *a, float *b, float *out, uint32_t i)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % cols;
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < shared; k++) {
            uint32_t vk = k;
            sum +=
                a[i * rows * shared + trow * shared +
                  vk] * b[i * shared * cols + vk * cols + tcol];
        }
        out[i * rows * cols + trow * cols + tcol] = sum;
    }
}

float
*Klas_GEMM_Batched_batched_matmul_f32(uint32_t batch,
                                      uint32_t rows,
                                      uint32_t shared,
                                      uint32_t cols, float *a, float *b)
{
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), batch * rows * cols);
    uint32_t idx = 0U;
    for (; idx < batch; idx++) {
        uint32_t i = idx;
        KPR_KCALL(__hoisted_batched_matmul_f32_0,
                  rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
                  1024U, 0U, rows, shared, cols, a, b, out, i);
        MUST(cudaDeviceSynchronize());
    }
    return out;
}
