
#include "Kuiper_MatMulTranspose.h"

__global__
/**
  hoisted when extracting matmul_transpose_gpu_f32_ff
*/
static void __hoisted_0(uint32_t m, uint32_t n, uint32_t k, float *gA,
                        float *gB, float *gC)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    float sum = 0.0f;
    for (; k1 < k; k1++)
        sum += gA[trow * k + k1] * gB[k1 * n + tcol];
    gC[tcol * m + trow] = sum;
}

/**
An example of computing tr(AB) by just shifting a view. Basically:
  - Instantiating rA=rB=row_major, rC=col_major
  - Do the product, we get C = AB (in col-major)
  - View-shift C to get tr(AB) in row-major

TODO: It would be nicer to do this just over a CPU-side matmul, but there is no
view-like interface for CPU arrays.
*/
void
Kuiper_MatMulTranspose_matmul_transpose_gpu_f32_ff(uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_KCALL(__hoisted_0, m * n, 1U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
