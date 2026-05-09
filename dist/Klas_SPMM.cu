
#include "Klas_SPMM.h"

__global__
/**
  hoisted when extracting spmm_u32
*/
static void
__hoisted_0(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
            uint32_t *row_indices, uint32_t *gB, uint32_t *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    uint32_t *elems_tile = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t out[4U] = { 0U };
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        uint32_t a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_spmm_u32(uint32_t rows,
                   uint32_t shared,
                   uint32_t cols,
                   Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                   uint32_t *row_indices, uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_0, rows * ((cols + 128U - 1U) / 128U), 32U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x16x16
*/
static void
__hoisted_1(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 16U - 1U) / 16U)];
    uint32_t n_idx = blockIdx.x % ((cols + 16U - 1U) / 16U) * 16U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 16U; nnz -= 16U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 16U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 16U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 16U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 16U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_16x16x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(128U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 128U));
    KPR_KCALL(__hoisted_1, rows * ((cols + 16U - 1U) / 16U), 16U, 128U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x32x16
*/
static void
__hoisted_2(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 16U - 1U) / 16U)];
    uint32_t n_idx = blockIdx.x % ((cols + 16U - 1U) / 16U) * 16U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_16x32x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_2, rows * ((cols + 16U - 1U) / 16U), 16U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x64x16
*/
static void
__hoisted_3(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 16U - 1U) / 16U)];
    uint32_t n_idx = blockIdx.x % ((cols + 16U - 1U) / 16U) * 16U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_16x64x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_3, rows * ((cols + 16U - 1U) / 16U), 16U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x128x16
*/
static void
__hoisted_4(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 16U - 1U) / 16U)];
    uint32_t n_idx = blockIdx.x % ((cols + 16U - 1U) / 16U) * 16U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_16x128x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_4, rows * ((cols + 16U - 1U) / 16U), 16U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x256x16
*/
static void
__hoisted_5(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 16U - 1U) / 16U)];
    uint32_t n_idx = blockIdx.x % ((cols + 16U - 1U) / 16U) * 16U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_16x256x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_5, rows * ((cols + 16U - 1U) / 16U), 16U, 2048U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x512x16
*/
static void
__hoisted_6(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 16U - 1U) / 16U)];
    uint32_t n_idx = blockIdx.x % ((cols + 16U - 1U) / 16U) * 16U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_16x512x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_6, rows * ((cols + 16U - 1U) / 16U), 16U, 4096U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x16x16
*/
static void
__hoisted_7(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 16U; nnz -= 16U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 16U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 16U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 16U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 16U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_32x16x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(128U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 128U));
    KPR_KCALL(__hoisted_7, rows * ((cols + 32U - 1U) / 32U), 16U, 128U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x32x16
*/
static void
__hoisted_8(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_32x32x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_8, rows * ((cols + 32U - 1U) / 32U), 16U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x32x32
*/
static void
__hoisted_9(uint32_t cols,
            Kuiper_Sparse_Matrix_smatrix__float gA,
            uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_32x32x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_9, rows * ((cols + 32U - 1U) / 32U), 32U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x64x16
*/
static void
__hoisted_10(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_32x64x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_10, rows * ((cols + 32U - 1U) / 32U), 16U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x64x32
*/
static void
__hoisted_11(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_32x64x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_11, rows * ((cols + 32U - 1U) / 32U), 32U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x128x16
*/
static void
__hoisted_12(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_32x128x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_12, rows * ((cols + 32U - 1U) / 32U), 16U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x128x32
*/
static void
__hoisted_13(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_32x128x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_13, rows * ((cols + 32U - 1U) / 32U), 32U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x256x16
*/
static void
__hoisted_14(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_32x256x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_14, rows * ((cols + 32U - 1U) / 32U), 16U, 2048U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x256x32
*/
static void
__hoisted_15(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_32x256x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_15, rows * ((cols + 32U - 1U) / 32U), 32U, 2048U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x512x16
*/
static void
__hoisted_16(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_32x512x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_16, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_16, rows * ((cols + 32U - 1U) / 32U), 16U, 4096U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x512x32
*/
static void
__hoisted_17(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 32U - 1U) / 32U)];
    uint32_t n_idx = blockIdx.x % ((cols + 32U - 1U) / 32U) * 32U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_32x512x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_17, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_17, rows * ((cols + 32U - 1U) / 32U), 32U, 4096U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x16x16
*/
static void
__hoisted_18(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 16U; nnz -= 16U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 16U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 16U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 16U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 16U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x16x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(128U);
    MUST(cudaFuncSetAttribute
         (__hoisted_18, cudaFuncAttributeMaxDynamicSharedMemorySize, 128U));
    KPR_KCALL(__hoisted_18, rows * ((cols + 64U - 1U) / 64U), 16U, 128U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x32x16
*/
static void
__hoisted_19(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x32x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_19, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_19, rows * ((cols + 64U - 1U) / 64U), 16U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x32x32
*/
static void
__hoisted_20(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x32x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_20, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_20, rows * ((cols + 64U - 1U) / 64U), 32U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x16
*/
static void
__hoisted_21(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x64x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_21, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_21, rows * ((cols + 64U - 1U) / 64U), 16U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x32
*/
static void
__hoisted_22(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x64x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_22, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_22, rows * ((cols + 64U - 1U) / 64U), 32U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x64
*/
static void
__hoisted_23(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_64x64x64(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_23, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_23, rows * ((cols + 64U - 1U) / 64U), 64U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x16
*/
static void
__hoisted_24(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x128x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_24, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_24, rows * ((cols + 64U - 1U) / 64U), 16U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x32
*/
static void
__hoisted_25(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x128x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_25, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_25, rows * ((cols + 64U - 1U) / 64U), 32U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x64
*/
static void
__hoisted_26(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_64x128x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_26, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_26, rows * ((cols + 64U - 1U) / 64U), 64U, 1024U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x16
*/
static void
__hoisted_27(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x256x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_27, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_27, rows * ((cols + 64U - 1U) / 64U), 16U, 2048U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x32
*/
static void
__hoisted_28(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x256x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_28, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_28, rows * ((cols + 64U - 1U) / 64U), 32U, 2048U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x64
*/
static void
__hoisted_29(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_64x256x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_29, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_29, rows * ((cols + 64U - 1U) / 64U), 64U, 2048U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x16
*/
static void
__hoisted_30(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x512x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_30, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_30, rows * ((cols + 64U - 1U) / 64U), 16U, 4096U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x32
*/
static void
__hoisted_31(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_64x512x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_31, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_31, rows * ((cols + 64U - 1U) / 64U), 32U, 4096U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x64
*/
static void
__hoisted_32(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 64U - 1U) / 64U)];
    uint32_t n_idx = blockIdx.x % ((cols + 64U - 1U) / 64U) * 64U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_64x512x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_32, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_32, rows * ((cols + 64U - 1U) / 64U), 64U, 4096U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x16x16
*/
static void
__hoisted_33(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 16U; nnz -= 16U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 16U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 16U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 16U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 16U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x16x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(128U);
    MUST(cudaFuncSetAttribute
         (__hoisted_33, cudaFuncAttributeMaxDynamicSharedMemorySize, 128U));
    KPR_KCALL(__hoisted_33, rows * ((cols + 128U - 1U) / 128U), 16U, 128U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x32x16
*/
static void
__hoisted_34(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x32x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_34, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_34, rows * ((cols + 128U - 1U) / 128U), 16U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x32x32
*/
static void
__hoisted_35(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x32x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_35, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_35, rows * ((cols + 128U - 1U) / 128U), 32U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x16
*/
static void
__hoisted_36(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x64x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_36, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_36, rows * ((cols + 128U - 1U) / 128U), 16U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x32
*/
static void
__hoisted_37(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x64x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_37, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_37, rows * ((cols + 128U - 1U) / 128U), 32U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x64
*/
static void
__hoisted_38(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x64x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_38, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_38, rows * ((cols + 128U - 1U) / 128U), 64U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x16
*/
static void
__hoisted_39(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x128x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_39, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_39, rows * ((cols + 128U - 1U) / 128U), 16U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x32
*/
static void
__hoisted_40(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x128x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_40, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_40, rows * ((cols + 128U - 1U) / 128U), 32U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x64
*/
static void
__hoisted_41(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x128x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_41, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_41, rows * ((cols + 128U - 1U) / 128U), 64U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x128
*/
static void
__hoisted_42(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_128x128x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_42, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_42, rows * ((cols + 128U - 1U) / 128U), 128U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x16
*/
static void
__hoisted_43(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x256x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_43, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_43, rows * ((cols + 128U - 1U) / 128U), 16U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x32
*/
static void
__hoisted_44(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x256x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_44, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_44, rows * ((cols + 128U - 1U) / 128U), 32U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x64
*/
static void
__hoisted_45(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x256x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_45, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_45, rows * ((cols + 128U - 1U) / 128U), 64U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x128
*/
static void
__hoisted_46(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_128x256x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_46, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_46, rows * ((cols + 128U - 1U) / 128U), 128U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x16
*/
static void
__hoisted_47(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x512x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_47, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_47, rows * ((cols + 128U - 1U) / 128U), 16U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x32
*/
static void
__hoisted_48(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x512x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_48, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_48, rows * ((cols + 128U - 1U) / 128U), 32U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x64
*/
static void
__hoisted_49(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_128x512x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_49, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_49, rows * ((cols + 128U - 1U) / 128U), 64U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x128
*/
static void
__hoisted_50(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 128U - 1U) / 128U)];
    uint32_t n_idx = blockIdx.x % ((cols + 128U - 1U) / 128U) * 128U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_128x512x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_50, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_50, rows * ((cols + 128U - 1U) / 128U), 128U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x16x16
*/
static void
__hoisted_51(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 16U; nnz -= 16U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 16U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 16U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 16U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 16U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x16x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(128U);
    MUST(cudaFuncSetAttribute
         (__hoisted_51, cudaFuncAttributeMaxDynamicSharedMemorySize, 128U));
    KPR_KCALL(__hoisted_51, rows * ((cols + 256U - 1U) / 256U), 16U, 128U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x32x16
*/
static void
__hoisted_52(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x32x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_52, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_52, rows * ((cols + 256U - 1U) / 256U), 16U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x32x32
*/
static void
__hoisted_53(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x32x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_53, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_53, rows * ((cols + 256U - 1U) / 256U), 32U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x16
*/
static void
__hoisted_54(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x64x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_54, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_54, rows * ((cols + 256U - 1U) / 256U), 16U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x32
*/
static void
__hoisted_55(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x64x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_55, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_55, rows * ((cols + 256U - 1U) / 256U), 32U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x64
*/
static void
__hoisted_56(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x64x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_56, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_56, rows * ((cols + 256U - 1U) / 256U), 64U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x16
*/
static void
__hoisted_57(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x128x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_57, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_57, rows * ((cols + 256U - 1U) / 256U), 16U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x32
*/
static void
__hoisted_58(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x128x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_58, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_58, rows * ((cols + 256U - 1U) / 256U), 32U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x64
*/
static void
__hoisted_59(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x128x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_59, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_59, rows * ((cols + 256U - 1U) / 256U), 64U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x128
*/
static void
__hoisted_60(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x128x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_60, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_60, rows * ((cols + 256U - 1U) / 256U), 128U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x16
*/
static void
__hoisted_61(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x256x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_61, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_61, rows * ((cols + 256U - 1U) / 256U), 16U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x32
*/
static void
__hoisted_62(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x256x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_62, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_62, rows * ((cols + 256U - 1U) / 256U), 32U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x64
*/
static void
__hoisted_63(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x256x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_63, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_63, rows * ((cols + 256U - 1U) / 256U), 64U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x128
*/
static void
__hoisted_64(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x256x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_64, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_64, rows * ((cols + 256U - 1U) / 256U), 128U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x256
*/
static void
__hoisted_65(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 256U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 255U - threadIdx.x) / 256U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 256U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 256U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 256U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_256x256x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_65, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_65, rows * ((cols + 256U - 1U) / 256U), 256U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x16
*/
static void
__hoisted_66(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x512x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_66, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_66, rows * ((cols + 256U - 1U) / 256U), 16U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x32
*/
static void
__hoisted_67(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x512x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_67, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_67, rows * ((cols + 256U - 1U) / 256U), 32U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x64
*/
static void
__hoisted_68(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x512x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_68, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_68, rows * ((cols + 256U - 1U) / 256U), 64U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x128
*/
static void
__hoisted_69(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_256x512x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_69, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_69, rows * ((cols + 256U - 1U) / 256U), 128U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x256
*/
static void
__hoisted_70(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 256U - 1U) / 256U)];
    uint32_t n_idx = blockIdx.x % ((cols + 256U - 1U) / 256U) * 256U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 256U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 255U - threadIdx.x) / 256U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 256U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 256U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 256U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_256x512x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_70, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_70, rows * ((cols + 256U - 1U) / 256U), 256U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x16x16
*/
static void
__hoisted_71(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 16U; nnz -= 16U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 16U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 16U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 32U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 16U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 16U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 32U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 32U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x16x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(128U);
    MUST(cudaFuncSetAttribute
         (__hoisted_71, cudaFuncAttributeMaxDynamicSharedMemorySize, 128U));
    KPR_KCALL(__hoisted_71, rows * ((cols + 512U - 1U) / 512U), 16U, 128U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x32x16
*/
static void
__hoisted_72(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 32U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 32U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 32U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x32x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_72, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_72, rows * ((cols + 512U - 1U) / 512U), 16U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x32x32
*/
static void
__hoisted_73(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 32U; nnz -= 32U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 32U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 32U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 32U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x32x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    MUST(cudaFuncSetAttribute
         (__hoisted_73, cudaFuncAttributeMaxDynamicSharedMemorySize, 256U));
    KPR_KCALL(__hoisted_73, rows * ((cols + 512U - 1U) / 512U), 32U, 256U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x16
*/
static void
__hoisted_74(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 32U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 32U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 32U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x64x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_74, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_74, rows * ((cols + 512U - 1U) / 512U), 16U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x32
*/
static void
__hoisted_75(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x64x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_75, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_75, rows * ((cols + 512U - 1U) / 512U), 32U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x64
*/
static void
__hoisted_76(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 64U; nnz -= 64U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 64U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 64U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 64U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x64x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    MUST(cudaFuncSetAttribute
         (__hoisted_76, cudaFuncAttributeMaxDynamicSharedMemorySize, 512U));
    KPR_KCALL(__hoisted_76, rows * ((cols + 512U - 1U) / 512U), 64U, 512U, cols,
              gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x16
*/
static void
__hoisted_77(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 32U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 32U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 32U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x128x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_77, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_77, rows * ((cols + 512U - 1U) / 512U), 16U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x32
*/
static void
__hoisted_78(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x128x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_78, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_78, rows * ((cols + 512U - 1U) / 512U), 32U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x64
*/
static void
__hoisted_79(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x128x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_79, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_79, rows * ((cols + 512U - 1U) / 512U), 64U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x128
*/
static void
__hoisted_80(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 128U; nnz -= 128U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 128U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 128U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 128U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x128x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    MUST(cudaFuncSetAttribute
         (__hoisted_80, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024U));
    KPR_KCALL(__hoisted_80, rows * ((cols + 512U - 1U) / 512U), 128U, 1024U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x16
*/
static void
__hoisted_81(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 32U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 32U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 32U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x256x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_81, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_81, rows * ((cols + 512U - 1U) / 512U), 16U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x32
*/
static void
__hoisted_82(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x256x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_82, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_82, rows * ((cols + 512U - 1U) / 512U), 32U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x64
*/
static void
__hoisted_83(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x256x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_83, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_83, rows * ((cols + 512U - 1U) / 512U), 64U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x128
*/
static void
__hoisted_84(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x256x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_84, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_84, rows * ((cols + 512U - 1U) / 512U), 128U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x256
*/
static void
__hoisted_85(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 256U; nnz -= 256U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 256U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 256U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 256U;
    __syncthreads();
    uint32_t tresidue = (re - off + 255U - threadIdx.x) / 256U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 256U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 256U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 256U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 256U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x256x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_85, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_85, rows * ((cols + 512U - 1U) / 512U), 256U, 2048U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x16
*/
static void
__hoisted_86(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t tile_off = i * 16U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 32U) {
                uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 15U - threadIdx.x) / 16U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 16U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 32U) {
            uint32_t dense_off = n_idx + x * 16U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 32U; i0++)
        if (n_idx + i0 * 16U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 16U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x512x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_86, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_86, rows * ((cols + 512U - 1U) / 512U), 16U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x32
*/
static void
__hoisted_87(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t tile_off = i * 32U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 16U) {
                uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 31U - threadIdx.x) / 32U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 32U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 16U) {
            uint32_t dense_off = n_idx + x * 32U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 16U; i0++)
        if (n_idx + i0 * 32U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 32U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x512x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_87, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_87, rows * ((cols + 512U - 1U) / 512U), 32U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x64
*/
static void
__hoisted_88(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8U; i++) {
            uint32_t tile_off = i * 64U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 8U) {
                uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 63U - threadIdx.x) / 64U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 64U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 8U) {
            uint32_t dense_off = n_idx + x * 64U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 64U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 64U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x512x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_88, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_88, rows * ((cols + 512U - 1U) / 512U), 64U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x128
*/
static void
__hoisted_89(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4U; i++) {
            uint32_t tile_off = i * 128U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 4U) {
                uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 127U - threadIdx.x) / 128U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 128U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 4U) {
            uint32_t dense_off = n_idx + x * 128U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 128U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 128U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x512x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_89, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_89, rows * ((cols + 512U - 1U) / 512U), 128U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x256
*/
static void
__hoisted_90(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2U; i++) {
            uint32_t tile_off = i * 256U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 2U) {
                uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
                if (dense_off < cols) {
                    out[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 255U - threadIdx.x) / 256U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 256U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 2U) {
            uint32_t dense_off = n_idx + x * 256U + threadIdx.x;
            if (dense_off < cols) {
                out[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 256U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 256U + threadIdx.x] = out[i0];
}

void
Klas_SPMM_g_spmm_f32_512x512x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_90, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_90, rows * ((cols + 512U - 1U) / 512U), 256U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x512
*/
static void
__hoisted_91(uint32_t cols,
             Kuiper_Sparse_Matrix_smatrix__float gA,
             uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x / ((cols + 512U - 1U) / 512U)];
    uint32_t n_idx = blockIdx.x % ((cols + 512U - 1U) / 512U) * 512U;
    float *elems_tile = (float *)KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(2048U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    float out = 0.0f;
    uint32_t nnz = re - ri;
    uint32_t idx = 0U;
    for (; nnz >= 512U; nnz -= 512U) {
        uint32_t __anf0 = idx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1U; i++) {
            uint32_t tile_off = i * 512U + threadIdx.x;
            uint32_t off1 = ri + __anf0 * 512U;
            elems_tile[tile_off] = gA.elems[off1 + tile_off];
            col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
        }
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            float a = elems_tile[k];
            uint32_t c = col_ind_tile[k];
            uint32_t x = 0U;
            while (x < 1U) {
                uint32_t dense_off = n_idx + x * 512U + threadIdx.x;
                if (dense_off < cols) {
                    (&out)[x] += a * gB[c * cols + dense_off];
                    x++;
                } else
                    x++;
            }
        }
        idx++;
    }
    uint32_t __anf0 = idx;
    uint32_t off = ri + __anf0 * 512U;
    __syncthreads();
    uint32_t tresidue = (re - off + 511U - threadIdx.x) / 512U;
    uint32_t i = 0U;
    for (; i < tresidue; i++) {
        uint32_t tile_off = i * 512U + threadIdx.x;
        uint32_t off1 = ri + __anf0 * 512U;
        elems_tile[tile_off] = gA.elems[off1 + tile_off];
        col_ind_tile[tile_off] = gA.col_ind[off1 + tile_off];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < nnz; k++) {
        float a = elems_tile[k];
        uint32_t c = col_ind_tile[k];
        uint32_t x = 0U;
        while (x < 1U) {
            uint32_t dense_off = n_idx + x * 512U + threadIdx.x;
            if (dense_off < cols) {
                (&out)[x] += a * gB[c * cols + dense_off];
                x++;
            } else
                x++;
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 512U + threadIdx.x < cols)
            gC[m_idx * cols + n_idx + i0 * 512U + threadIdx.x] = (&out)[i0];
}

void
Klas_SPMM_g_spmm_f32_512x512x512(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_91, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_91, rows * ((cols + 512U - 1U) / 512U), 512U, 4096U,
              cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}
