
#include "Klas_SPMM.h"

__global__
/**
  hoisted when extracting spmm_u32
*/
static void
__hoisted_spmm_u32_0(uint32_t cols,
                     Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                     uint32_t *row_indices, uint32_t *gB, uint32_t *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_spmm_u32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_spmm_u32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              32U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting spmm_f32
*/
static void
__hoisted_spmm_f32_0(uint32_t cols,
                     Kuiper_Sparse_Matrix_smatrix__float gA,
                     uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
Klas_SPMM_spmm_f32(uint32_t rows,
                   uint32_t shared,
                   uint32_t cols,
                   Kuiper_Sparse_Matrix_smatrix__float gA,
                   uint32_t *row_indices, float *gB, float *gC)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_spmm_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_spmm_f32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              64U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x16x16
*/
static void
__hoisted_g_spmm_f32_16x16x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 16U + (uint32_t) (cols % 16U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 16U + (uint32_t) (cols % 16U != 0U)) * 16U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_16x16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              128U));
    KPR_KCALL(__hoisted_g_spmm_f32_16x16x16_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)),
              16U, 128U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x32x16
*/
static void
__hoisted_g_spmm_f32_16x32x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 16U + (uint32_t) (cols % 16U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 16U + (uint32_t) (cols % 16U != 0U)) * 16U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_16x32x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_16x32x16_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)),
              16U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x64x16
*/
static void
__hoisted_g_spmm_f32_16x64x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 16U + (uint32_t) (cols % 16U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 16U + (uint32_t) (cols % 16U != 0U)) * 16U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_16x64x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_16x64x16_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)),
              16U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x128x16
*/
static void
__hoisted_g_spmm_f32_16x128x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 16U + (uint32_t) (cols % 16U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 16U + (uint32_t) (cols % 16U != 0U)) * 16U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_16x128x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_16x128x16_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)),
              16U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x256x16
*/
static void
__hoisted_g_spmm_f32_16x256x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 16U + (uint32_t) (cols % 16U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 16U + (uint32_t) (cols % 16U != 0U)) * 16U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_16x256x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_16x256x16_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)),
              16U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_16x512x16
*/
static void
__hoisted_g_spmm_f32_16x512x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 16U + (uint32_t) (cols % 16U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 16U + (uint32_t) (cols % 16U != 0U)) * 16U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_16x512x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_16x512x16_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)),
              16U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x16x16
*/
static void
__hoisted_g_spmm_f32_32x16x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              128U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x16x16_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              16U, 128U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x32x16
*/
static void
__hoisted_g_spmm_f32_32x32x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x32x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x32x16_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              16U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x32x32
*/
static void
__hoisted_g_spmm_f32_32x32x32_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x32x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x32x32_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              32U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x64x16
*/
static void
__hoisted_g_spmm_f32_32x64x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x64x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x64x16_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              16U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x64x32
*/
static void
__hoisted_g_spmm_f32_32x64x32_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x64x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x64x32_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              32U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x128x16
*/
static void
__hoisted_g_spmm_f32_32x128x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x128x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x128x16_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              16U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x128x32
*/
static void
__hoisted_g_spmm_f32_32x128x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x128x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x128x32_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              32U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x256x16
*/
static void
__hoisted_g_spmm_f32_32x256x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x256x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x256x16_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              16U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x256x32
*/
static void
__hoisted_g_spmm_f32_32x256x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x256x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x256x32_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              32U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x512x16
*/
static void
__hoisted_g_spmm_f32_32x512x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x512x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x512x16_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              16U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x512x32
*/
static void
__hoisted_g_spmm_f32_32x512x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 32U + (uint32_t) (cols % 32U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 32U + (uint32_t) (cols % 32U != 0U)) * 32U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x512x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x512x32_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)),
              32U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x16x16
*/
static void
__hoisted_g_spmm_f32_64x16x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              128U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x16x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              16U, 128U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x32x16
*/
static void
__hoisted_g_spmm_f32_64x32x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x32x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x32x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              16U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x32x32
*/
static void
__hoisted_g_spmm_f32_64x32x32_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x32x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x32x32_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              32U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x16
*/
static void
__hoisted_g_spmm_f32_64x64x16_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x64x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              16U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x32
*/
static void
__hoisted_g_spmm_f32_64x64x32_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x64x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x32_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              32U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x64
*/
static void
__hoisted_g_spmm_f32_64x64x64_0(uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x64x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x64_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              64U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x16
*/
static void
__hoisted_g_spmm_f32_64x128x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x128x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              16U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x32
*/
static void
__hoisted_g_spmm_f32_64x128x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x128x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x32_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              32U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x64
*/
static void
__hoisted_g_spmm_f32_64x128x64_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x128x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x64_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              64U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x16
*/
static void
__hoisted_g_spmm_f32_64x256x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x256x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              16U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x32
*/
static void
__hoisted_g_spmm_f32_64x256x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x256x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x32_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              32U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x64
*/
static void
__hoisted_g_spmm_f32_64x256x64_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x256x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x64_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              64U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x16
*/
static void
__hoisted_g_spmm_f32_64x512x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x512x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              16U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x32
*/
static void
__hoisted_g_spmm_f32_64x512x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x512x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x32_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              32U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x64
*/
static void
__hoisted_g_spmm_f32_64x512x64_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x / (cols / 64U + (uint32_t) (cols % 64U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 64U + (uint32_t) (cols % 64U != 0U)) * 64U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x512x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x64_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)),
              64U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x16x16
*/
static void
__hoisted_g_spmm_f32_128x16x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x16x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              16U, 128U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x32x16
*/
static void
__hoisted_g_spmm_f32_128x32x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x32x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x32x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              16U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x32x32
*/
static void
__hoisted_g_spmm_f32_128x32x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x32x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x32x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              32U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x16
*/
static void
__hoisted_g_spmm_f32_128x64x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x64x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              16U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x32
*/
static void
__hoisted_g_spmm_f32_128x64x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x64x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              32U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x64
*/
static void
__hoisted_g_spmm_f32_128x64x64_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x64x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x64_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              64U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x16
*/
static void
__hoisted_g_spmm_f32_128x128x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              16U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x32
*/
static void
__hoisted_g_spmm_f32_128x128x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              32U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x64
*/
static void
__hoisted_g_spmm_f32_128x128x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x64_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              64U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x128
*/
static void
__hoisted_g_spmm_f32_128x128x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x128_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              128U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x16
*/
static void
__hoisted_g_spmm_f32_128x256x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              16U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x32
*/
static void
__hoisted_g_spmm_f32_128x256x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              32U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x64
*/
static void
__hoisted_g_spmm_f32_128x256x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x64_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              64U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x128
*/
static void
__hoisted_g_spmm_f32_128x256x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x128_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              128U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x16
*/
static void
__hoisted_g_spmm_f32_128x512x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              16U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x32
*/
static void
__hoisted_g_spmm_f32_128x512x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              32U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x64
*/
static void
__hoisted_g_spmm_f32_128x512x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x64_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              64U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x128
*/
static void
__hoisted_g_spmm_f32_128x512x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x128_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)),
              128U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x16x16
*/
static void
__hoisted_g_spmm_f32_256x16x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              128U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x16x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              16U, 128U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x32x16
*/
static void
__hoisted_g_spmm_f32_256x32x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x32x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x32x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              16U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x32x32
*/
static void
__hoisted_g_spmm_f32_256x32x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x32x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x32x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              32U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x16
*/
static void
__hoisted_g_spmm_f32_256x64x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x64x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              16U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x32
*/
static void
__hoisted_g_spmm_f32_256x64x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x64x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              32U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x64
*/
static void
__hoisted_g_spmm_f32_256x64x64_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x64x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x64_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              64U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x16
*/
static void
__hoisted_g_spmm_f32_256x128x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              16U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x32
*/
static void
__hoisted_g_spmm_f32_256x128x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              32U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x64
*/
static void
__hoisted_g_spmm_f32_256x128x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x64_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              64U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x128
*/
static void
__hoisted_g_spmm_f32_256x128x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x128_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              128U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x16
*/
static void
__hoisted_g_spmm_f32_256x256x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              16U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x32
*/
static void
__hoisted_g_spmm_f32_256x256x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              32U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x64
*/
static void
__hoisted_g_spmm_f32_256x256x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x64_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              64U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x128
*/
static void
__hoisted_g_spmm_f32_256x256x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x128_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              128U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x256
*/
static void
__hoisted_g_spmm_f32_256x256x256_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x256_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x256_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              256U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x16
*/
static void
__hoisted_g_spmm_f32_256x512x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              16U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x32
*/
static void
__hoisted_g_spmm_f32_256x512x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              32U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x64
*/
static void
__hoisted_g_spmm_f32_256x512x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x64_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              64U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x128
*/
static void
__hoisted_g_spmm_f32_256x512x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x128_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              128U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x256
*/
static void
__hoisted_g_spmm_f32_256x512x256_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 256U + (uint32_t) (cols % 256U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 256U + (uint32_t) (cols % 256U != 0U)) * 256U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x256_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x256_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)),
              256U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x16x16
*/
static void
__hoisted_g_spmm_f32_512x16x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              128U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x16x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              16U, 128U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x32x16
*/
static void
__hoisted_g_spmm_f32_512x32x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x32x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x32x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              16U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x32x32
*/
static void
__hoisted_g_spmm_f32_512x32x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x32x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              256U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x32x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              32U, 256U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x16
*/
static void
__hoisted_g_spmm_f32_512x64x16_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x64x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              16U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x32
*/
static void
__hoisted_g_spmm_f32_512x64x32_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x64x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              32U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x64
*/
static void
__hoisted_g_spmm_f32_512x64x64_0(uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x64x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x64_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              64U, 512U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x16
*/
static void
__hoisted_g_spmm_f32_512x128x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              16U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x32
*/
static void
__hoisted_g_spmm_f32_512x128x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              32U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x64
*/
static void
__hoisted_g_spmm_f32_512x128x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x64_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              64U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x128
*/
static void
__hoisted_g_spmm_f32_512x128x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              1024U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x128_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              128U, 1024U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x16
*/
static void
__hoisted_g_spmm_f32_512x256x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              16U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x32
*/
static void
__hoisted_g_spmm_f32_512x256x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              32U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x64
*/
static void
__hoisted_g_spmm_f32_512x256x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x64_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              64U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x128
*/
static void
__hoisted_g_spmm_f32_512x256x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x128_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              128U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x256
*/
static void
__hoisted_g_spmm_f32_512x256x256_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x256_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x256_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              256U, 2048U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x16
*/
static void
__hoisted_g_spmm_f32_512x512x16_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              16U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x32
*/
static void
__hoisted_g_spmm_f32_512x512x32_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              32U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x64
*/
static void
__hoisted_g_spmm_f32_512x512x64_0(uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x64_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              64U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x128
*/
static void
__hoisted_g_spmm_f32_512x512x128_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x128_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x128_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              128U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x256
*/
static void
__hoisted_g_spmm_f32_512x512x256_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x256_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x256_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              256U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x512
*/
static void
__hoisted_g_spmm_f32_512x512x512_0(uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 512U + (uint32_t) (cols % 512U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 512U + (uint32_t) (cols % 512U != 0U)) * 512U;
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
    uint32_t __anf01 = nnz;
    uint32_t k = 0U;
    for (; k < __anf01; k++) {
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
    MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x512_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x512_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)),
              512U, 4096U, cols, gA, row_indices, gB, gC);
    MUST(cudaDeviceSynchronize());
}
