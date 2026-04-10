
#include "Kuiper_Sparse_GEMM.h"

__global__
/**
  hoisted when extracting _gemm_u32_rr
*/
static void
__hoisted_0(uint32_t rows,
            uint32_t cols,
            Kuiper_Sparse_smatrix__uint32_t gA, uint32_t *gB, uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % cols;
        uint32_t dp = 0U;
        uint32_t k = gA.row_off[trow];
        for (; k < gA.row_off[trow + 1U]; k++)
            dp += gA.elems1[k] * gB[gA.col_ind[k] * cols + tcol];
        gC[trow * cols + tcol] = dp;
    }
}

void
Kuiper_Sparse_GEMM__gemm_u32_rr(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_smatrix__uint32_t gA,
                                uint32_t *gB, uint32_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, (rows * cols + 1023U) / 1024U, 1024U, 0U, rows, cols,
              gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
