
#include "Kuiper_MatMulTranspose.h"

__global__
/**
  hoisted when extracting matmul_transpose_gpu_f32_ff
*/
static void
__hoisted_0(uint32_t rows, uint32_t shared, uint32_t cols, float *gA, float *gB,
            float *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[tcol * rows + trow] = sum;
}

void
Kuiper_MatMulTranspose_matmul_transpose_gpu_f32_ff(uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, rows * cols, 1U, 0U, rows, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
