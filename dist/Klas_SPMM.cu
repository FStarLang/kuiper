
#include "Klas_SPMM.h"

typedef struct parameters_s {
    uint32_t rows;
    uint32_t shared;
    uint32_t cols;
    uint32_t blockItemsK;
    uint32_t blockItemsX;
    uint32_t blockWidth;
} parameters;

__global__
/**
  hoisted when extracting spmm_u32
*/
static void
__hoisted_0(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
            uint32_t *row_indices,
            uint32_t *gB,
            uint32_t *gC,
            uint32_t params__rows,
            uint32_t params__shared,
            uint32_t params__cols,
            uint32_t params__blockItemsK,
            uint32_t params__blockItemsX, uint32_t params__blockWidth)
{
    KRML_MAYBE_UNUSED_VAR(params__rows);
    KRML_MAYBE_UNUSED_VAR(params__shared);
    uint32_t
        m_idx =
        row_indices[blockIdx.x /
                    ((params__cols + params__blockItemsX -
                      1U) / params__blockItemsX)];
    uint32_t n_idx =
        blockIdx.x % ((params__cols + params__blockItemsX - 1U) /
                      params__blockItemsX) * params__blockItemsX;
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t out[4U] = { 0U };
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= params__blockItemsK; nnz -= params__blockItemsK) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < params__blockItemsK / params__blockWidth; i++) {
            uint32_t tile_off = i * params__blockWidth + threadIdx.x;
            uint32_t off1 = ri + __anf0 * params__blockItemsK;
            uint32_t x1 = gA.elems[off1 + tile_off];
            ((uint32_t *) KPR_SHMEM_AT(0U))[tile_off] = x1;
            uint32_t c = gA.col_ind[off1 + tile_off];
            ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[tile_off] = c;
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < params__blockItemsK; k++) {
            uint32_t __anf01 = k;
            uint32_t a = ((uint32_t *) KPR_SHMEM_AT(0U))[__anf01];
            uint32_t __anf02 = k;
            uint32_t c =
                ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[__anf02];
            uint32_t x = 0U;
            while (x < params__blockItemsX / params__blockWidth) {
                uint32_t dense_off =
                    n_idx + x * params__blockWidth + threadIdx.x;
                if (dense_off < params__cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t residue = re - (ri + idx * params__blockItemsK);
    uint32_t __anf01 = idx;
    uint32_t off = ri + __anf01 * params__blockItemsK;
    __syncthreads();
    uint32_t tresidue =
        (re - off + params__blockWidth - 1U - threadIdx.x) / params__blockWidth;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * params__blockWidth + threadIdx.x;
        uint32_t off1 = ri + __anf01 * params__blockItemsK;
        uint32_t x = gA.elems[off1 + tile_off];
        ((uint32_t *) KPR_SHMEM_AT(0U))[tile_off] = x;
        uint32_t c = gA.col_ind[off1 + tile_off];
        ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[tile_off] = c;
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < residue; k++) {
        uint32_t __anf02 = k;
        uint32_t a = ((uint32_t *) KPR_SHMEM_AT(0U))[__anf02];
        uint32_t __anf03 = k;
        uint32_t c =
            ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[__anf03];
        uint32_t x = 0U;
        while (x < params__blockItemsX / params__blockWidth) {
            uint32_t dense_off = n_idx + x * params__blockWidth + threadIdx.x;
            if (dense_off < params__cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < params__blockItemsX / params__blockWidth; i0++)
        if (n_idx + i0 * params__blockWidth + threadIdx.x < params__cols)
            gC[m_idx * cols + n_idx + i0 * params__blockWidth + threadIdx.x] =
                out[i0];
}

void
Klas_SPMM_spmm_u32(uint32_t rows,
                   uint32_t shared,
                   uint32_t cols,
                   Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                   uint32_t *row_indices, uint32_t *gB, uint32_t *gC)
{
    parameters params = {
        .rows = rows,.shared = shared,.cols = cols,.blockItemsK =
            128U,.blockItemsX = 128U,
        .blockWidth = 32U
    };
    KPR_SHMEM_FITS(4U * params.blockItemsK + 4U * params.blockItemsK);
    MUST(cudaFuncSetAttribute(__hoisted_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * params.blockItemsK +
                              4U * params.blockItemsK));
    KPR_KCALL(__hoisted_0,
              params.rows * ((params.cols + params.blockItemsX - 1U) /
                             params.blockItemsX), params.blockWidth,
              4U * params.blockItemsK + 4U * params.blockItemsK, cols, gA,
              row_indices, gB, gC, params.rows, params.shared, params.cols,
              params.blockItemsK, params.blockItemsX, params.blockWidth);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting spmm_f32
*/
static void
__hoisted_1(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices,
            float *gB,
            float *gC,
            uint32_t params__rows,
            uint32_t params__shared,
            uint32_t params__cols,
            uint32_t params__blockItemsK,
            uint32_t params__blockItemsX, uint32_t params__blockWidth)
{
    KRML_MAYBE_UNUSED_VAR(params__rows);
    KRML_MAYBE_UNUSED_VAR(params__shared);
    uint32_t
        m_idx =
        row_indices[blockIdx.x /
                    ((params__cols + params__blockItemsX -
                      1U) / params__blockItemsX)];
    uint32_t n_idx =
        blockIdx.x % ((params__cols + params__blockItemsX - 1U) /
                      params__blockItemsX) * params__blockItemsX;
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= params__blockItemsK; nnz -= params__blockItemsK) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < params__blockItemsK / params__blockWidth; i++) {
            uint32_t tile_off = i * params__blockWidth + threadIdx.x;
            uint32_t off1 = ri + __anf0 * params__blockItemsK;
            float x1 = gA.elems[off1 + tile_off];
            ((float *)KPR_SHMEM_AT(0U))[tile_off] = x1;
            uint32_t c = gA.col_ind[off1 + tile_off];
            ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[tile_off] = c;
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < params__blockItemsK; k++) {
            uint32_t __anf01 = k;
            float a = ((float *)KPR_SHMEM_AT(0U))[__anf01];
            uint32_t __anf02 = k;
            uint32_t c =
                ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[__anf02];
            uint32_t x = 0U;
            while (x < params__blockItemsX / params__blockWidth) {
                uint32_t dense_off =
                    n_idx + x * params__blockWidth + threadIdx.x;
                if (dense_off < params__cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t residue = re - (ri + idx * params__blockItemsK);
    uint32_t __anf01 = idx;
    uint32_t off = ri + __anf01 * params__blockItemsK;
    __syncthreads();
    uint32_t tresidue =
        (re - off + params__blockWidth - 1U - threadIdx.x) / params__blockWidth;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * params__blockWidth + threadIdx.x;
        uint32_t off1 = ri + __anf01 * params__blockItemsK;
        float x = gA.elems[off1 + tile_off];
        ((float *)KPR_SHMEM_AT(0U))[tile_off] = x;
        uint32_t c = gA.col_ind[off1 + tile_off];
        ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[tile_off] = c;
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < residue; k++) {
        uint32_t __anf02 = k;
        float a = ((float *)KPR_SHMEM_AT(0U))[__anf02];
        uint32_t __anf03 = k;
        uint32_t c =
            ((uint32_t *) KPR_SHMEM_AT(4U * params__blockItemsK))[__anf03];
        uint32_t x = 0U;
        while (x < params__blockItemsX / params__blockWidth) {
            uint32_t dense_off = n_idx + x * params__blockWidth + threadIdx.x;
            if (dense_off < params__cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < params__blockItemsX / params__blockWidth; i0++)
        if (n_idx + i0 * params__blockWidth + threadIdx.x < params__cols)
            gC[m_idx * cols + n_idx + i0 * params__blockWidth + threadIdx.x] =
                out[i0];
}

void
Klas_SPMM_spmm_f32(uint32_t rows,
                   uint32_t shared,
                   uint32_t cols,
                   Kuiper_Sparse_Matrix_smatrix__float gA,
                   uint32_t *row_indices, float *gB, float *gC)
{
    parameters params = {
        .rows = rows,.shared = shared,.cols = cols,.blockItemsK =
            128U,.blockItemsX = 128U,
        .blockWidth = 32U
    };
    KPR_SHMEM_FITS(4U * params.blockItemsK + 4U * params.blockItemsK);
    MUST(cudaFuncSetAttribute(__hoisted_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * params.blockItemsK +
                              4U * params.blockItemsK));
    KPR_KCALL(__hoisted_1,
              params.rows * ((params.cols + params.blockItemsX - 1U) /
                             params.blockItemsX), params.blockWidth,
              4U * params.blockItemsK + 4U * params.blockItemsK, cols, gA,
              row_indices, gB, gC, params.rows, params.shared, params.cols,
              params.blockItemsK, params.blockItemsX, params.blockWidth);
    MUST(cudaDeviceSynchronize());
}
