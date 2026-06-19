
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
    uint32_t m_idx =
        row_indices[blockIdx.x /
                    (cols / 128U + (uint32_t) (cols % 128U != 0U))];
    uint32_t n_idx =
        blockIdx.x % (cols / 128U + (uint32_t) (cols % 128U != 0U)) * 128U;
    uint32_t *elems_tile = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    uint32_t out[4U] = { 0U };
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0U;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting spmm_f32
*/
static void
__hoisted_1(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_64x16x16
*/
static void
__hoisted_2(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 64U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_64x32x16
*/
static void
__hoisted_3(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x16
*/
static void
__hoisted_4(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x16
*/
static void
__hoisted_5(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x16
*/
static void
__hoisted_6(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x16
*/
static void
__hoisted_7(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x16x16
*/
static void
__hoisted_8(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x32x16
*/
static void
__hoisted_9(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x32x32
*/
static void
__hoisted_10(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x16
*/
static void
__hoisted_11(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x32
*/
static void
__hoisted_12(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x16
*/
static void
__hoisted_13(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x32
*/
static void
__hoisted_14(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x16
*/
static void
__hoisted_15(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x32
*/
static void
__hoisted_16(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x16
*/
static void
__hoisted_17(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x32
*/
static void
__hoisted_18(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x16x16
*/
static void
__hoisted_19(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x32x16
*/
static void
__hoisted_20(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x32x32
*/
static void
__hoisted_21(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x16
*/
static void
__hoisted_22(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x32
*/
static void
__hoisted_23(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x64
*/
static void
__hoisted_24(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x16
*/
static void
__hoisted_25(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x32
*/
static void
__hoisted_26(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x64
*/
static void
__hoisted_27(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x16
*/
static void
__hoisted_28(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x32
*/
static void
__hoisted_29(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x64
*/
static void
__hoisted_30(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x16
*/
static void
__hoisted_31(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x32
*/
static void
__hoisted_32(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x64
*/
static void
__hoisted_33(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x16x16
*/
static void
__hoisted_34(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x32x16
*/
static void
__hoisted_35(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x32x32
*/
static void
__hoisted_36(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x16
*/
static void
__hoisted_37(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x32
*/
static void
__hoisted_38(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x64
*/
static void
__hoisted_39(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x16
*/
static void
__hoisted_40(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x32
*/
static void
__hoisted_41(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x64
*/
static void
__hoisted_42(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x128
*/
static void
__hoisted_43(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out = 0.0f;
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 128U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 128U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 128U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 128U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 128U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 128U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 128U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 128U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 128U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 127U - threadIdx.x) / 128U; i++) {
        elems_tile[i * 128U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 128U + threadIdx.x];
        col_ind_tile[i * 128U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 128U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x16
*/
static void
__hoisted_44(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x32
*/
static void
__hoisted_45(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x64
*/
static void
__hoisted_46(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x128
*/
static void
__hoisted_47(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[2U];
    memset(out, 0U, 2U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 128U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 128U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 128U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 128U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 128U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 128U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 128U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 128U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 128U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 127U - threadIdx.x) / 128U; i++) {
        elems_tile[i * 128U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 128U + threadIdx.x];
        col_ind_tile[i * 128U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 128U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x16
*/
static void
__hoisted_48(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 * 16U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 16U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 16U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i * 16U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 16U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 16U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 15U - threadIdx.x) / 16U; i++) {
        elems_tile[i * 16U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 16U + threadIdx.x];
        col_ind_tile[i * 16U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 16U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x32
*/
static void
__hoisted_49(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 32U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 32U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 32U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 32U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 32U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 31U - threadIdx.x) / 32U; i++) {
        elems_tile[i * 32U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 32U + threadIdx.x];
        col_ind_tile[i * 32U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 32U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x64
*/
static void
__hoisted_50(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 64U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 64U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 64U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 64U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 64U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 64U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 63U - threadIdx.x) / 64U; i++) {
        elems_tile[i * 64U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 64U + threadIdx.x];
        col_ind_tile[i * 64U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 64U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x128
*/
static void
__hoisted_51(uint32_t cols,
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
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 128U + threadIdx.x) * 4U,
                       gA.elems + ri_ + (i0 * 128U + threadIdx.x) * 4U);
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 128U + threadIdx.x) * 4U,
                       gA.col_ind + ri_ + (i1 * 128U + threadIdx.x) * 4U);
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 128U + threadIdx.x] = 0.0f;
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
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 128U + threadIdx.x) * 4U,
                           gA.elems + off + (i * 128U + threadIdx.x) * 4U);
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 128U + threadIdx.x) * 4U,
                           gA.col_ind + off + (i0 * 128U + threadIdx.x) * 4U);
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
    } else {
        idx = 0U;
        nnz = re - ri;
    }
    uint32_t __anf0 = nnz;
    __syncthreads();
    uint32_t i = 0U;
    for (; i < (__anf0 + 127U - threadIdx.x) / 128U; i++) {
        elems_tile[i * 128U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 128U + threadIdx.x];
        col_ind_tile[i * 128U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 128U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
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
