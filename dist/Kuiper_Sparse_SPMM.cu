
#include "Kuiper_Sparse_SPMM.h"

__global__
/**
  hoisted when extracting _spmm_u32
*/
static void
__hoisted_0(uint32_t cols,
            Kuiper_Sparse_smatrix__uint32_t gA,
            uint32_t *gB, uint32_t *gC, uint32_t blockItemsK)
{
    uint32_t ri = gA.row_off[blockIdx.x];
    uint32_t dp = 0U;
    uint32_t nnz = gA.row_off[blockIdx.x + 1U] - ri;
    uint32_t idx = 0U;
    for (; nnz >= blockItemsK; nnz -= blockItemsK) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t x = gA.elems1[ri + __anf0 * blockItemsK + threadIdx.x];
        ((uint32_t *) KPR_SHMEM_AT(0U))[threadIdx.x] = x;
        uint32_t c = gA.col_ind[ri + __anf0 * blockItemsK + threadIdx.x];
        ((uint32_t *) KPR_SHMEM_AT(4U * blockItemsK))[threadIdx.x] = c;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < blockItemsK; k++) {
            uint32_t __anf01 = k;
            uint32_t x = ((uint32_t *) KPR_SHMEM_AT(0U))[__anf01];
            uint32_t __anf02 = k;
            uint32_t c = ((uint32_t *) KPR_SHMEM_AT(4U * blockItemsK))[__anf02];
            dp += x * gB[c * cols + threadIdx.x];
        }
        idx++;
    }
    gC[blockIdx.x * cols + threadIdx.x] = dp;
}

void
Kuiper_Sparse_SPMM__spmm_u32(uint32_t rows,
                             uint32_t shared,
                             uint32_t cols,
                             Kuiper_Sparse_smatrix__uint32_t gA,
                             uint32_t *gB, uint32_t *gC, uint32_t blockItemsK)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS(4U * blockItemsK + 4U * blockItemsK);
    MUST(cudaFuncSetAttribute(__hoisted_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * blockItemsK + 4U * blockItemsK));
    KPR_KCALL(__hoisted_0,
              rows,
              cols,
              4U * blockItemsK + 4U * blockItemsK,
              cols, gA, gB, gC, blockItemsK);
    MUST(cudaDeviceSynchronize());
}
