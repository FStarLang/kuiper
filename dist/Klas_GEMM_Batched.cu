
#include "Klas_GEMM_Batched.h"

__global__
/**
  hoisted when extracting batched_gemm_f32
*/
static void
__hoisted_batched_gemm_f32_0(uint32_t rows,
                             uint32_t shared,
                             uint32_t cols,
                             float *a, float *b, float *out, uint32_t i)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % cols;
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < shared; k++)
            sum +=
                a[i * (rows * shared) + trow * shared +
                  k] * b[i * (shared * cols) + k * cols + tcol];
        out[i * (rows * cols) + trow * cols + tcol] = sum;
    }
}

float
*Klas_GEMM_Batched_batched_gemm_f32(uint32_t batch,
                                    uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols, float *a, float *b)
{
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), batch * (rows * cols));
    uint32_t idx = 0U;
    for (; idx < batch; idx++) {
        KPR_KCALL(__hoisted_batched_gemm_f32_0,
                  rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
                  1024U, 0U, rows, shared, cols, a, b, out, idx);
        MUST(cudaDeviceSynchronize());
    }
    return out;
}
