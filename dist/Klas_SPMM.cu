
#include "Klas_SPMM.h"

__global__
/**
  hoisted when extracting spmm_u32
*/
static void
__hoisted_spmm_u32_0(uint32_t rows, uint32_t cols,
                     Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                     uint32_t *row_indices, uint32_t *gB, uint32_t *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    uint32_t *elems_tile = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    uint32_t out[4U] = {0U};
    if (nnz >= 128U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 32U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0U;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            uint32_t kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    uint32_t lchunk[4U] = {0U};
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                uint32_t kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        uint32_t lchunk[4U] = {0U};
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        uint32_t kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                uint32_t lchunk[4U] = {0U};
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_spmm_u32(uint32_t rows, uint32_t shared, uint32_t cols,
                        Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                        uint32_t *row_indices, uint32_t *gB, uint32_t *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_spmm_u32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting spmm_f32
*/
static void
__hoisted_spmm_f32_0(uint32_t rows, uint32_t cols,
                     Kuiper_Sparse_Matrix_smatrix__float gA,
                     uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_spmm_f32(uint32_t rows, uint32_t shared, uint32_t cols,
                        Kuiper_Sparse_Matrix_smatrix__float gA,
                        uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_spmm_f32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x4x1
*/
static void
__hoisted_g_spmm_f32_32x4x1_0(uint32_t rows, uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 4U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk, gB + (cols * kr + n_idx + __anf02 * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i + threadIdx.x) * 4U,
                           gA.elems + (off + (i + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                                   gB + (cols * kr + n_idx + __anf012 * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < __anf0 - threadIdx.x; i++) {
        elems_tile[i + threadIdx.x] = gA.elems[re - __anf0 + i + threadIdx.x];
        col_ind_tile[i + threadIdx.x] =
            gA.col_ind[re - __anf0 + i + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk, gB + (cols * kr + n_idx + __anf02 * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 4U), out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x4x1(uint32_t rows, uint32_t shared, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x4x1_0,
              rows * (cols / 4U + (uint32_t) (cols % 4U != 0U)), 1U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x8x2
*/
static void
__hoisted_g_spmm_f32_32x8x2_0(uint32_t rows, uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 8U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 2U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 2U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 2U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 2U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 2U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 2U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 2U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 2U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 2U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 2U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 2U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 2U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 2U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 1U - threadIdx.x) / 2U; i++) {
        elems_tile[i * 2U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 2U + threadIdx.x];
        col_ind_tile[i * 2U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 2U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 2U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 2U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 2U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 2U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x8x2(uint32_t rows, uint32_t shared, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x8x2_0,
              rows * (cols / 8U + (uint32_t) (cols % 8U != 0U)), 2U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x16x4
*/
static void
__hoisted_g_spmm_f32_32x16x4_0(uint32_t rows, uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 16U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 4U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 4U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 4U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 4U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 4U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 4U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 4U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 4U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 4U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 4U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 4U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 4U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 3U - threadIdx.x) / 4U; i++) {
        elems_tile[i * 4U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 4U + threadIdx.x];
        col_ind_tile[i * 4U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 4U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 4U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 4U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 4U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 4U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x16x4(uint32_t rows, uint32_t shared, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x16x4_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)), 4U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x32x8
*/
static void
__hoisted_g_spmm_f32_32x32x8_0(uint32_t rows, uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 8U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 8U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 8U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 8U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 8U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 8U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 8U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 8U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 8U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 8U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 7U - threadIdx.x) / 8U; i++) {
        elems_tile[i * 8U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 8U + threadIdx.x];
        col_ind_tile[i * 8U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 8U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 8U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 8U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x32x8(uint32_t rows, uint32_t shared, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x32x8_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 8U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x64x8
*/
static void
__hoisted_g_spmm_f32_32x64x8_0(uint32_t rows, uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 8U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 8U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 8U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 8U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 8U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 8U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 8U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 8U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 8U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 8U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 7U - threadIdx.x) / 8U; i++) {
        elems_tile[i * 8U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 8U + threadIdx.x];
        col_ind_tile[i * 8U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 8U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 8U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 8U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x64x8(uint32_t rows, uint32_t shared, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x64x8_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 8U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x4x1_on
*/
static void
__hoisted_g_spmm_f32_32x4x1_on_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 4U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 8U; i0++)
            vec_memcpy(elems_tile + (i0 + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk, gB + (cols * kr + n_idx + __anf02 * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 8U; i++)
                vec_memcpy(elems_tile + (i + threadIdx.x) * 4U,
                           gA.elems + (off + (i + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                                   gB + (cols * kr + n_idx + __anf012 * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < __anf0 - threadIdx.x; i++) {
        elems_tile[i + threadIdx.x] = gA.elems[re - __anf0 + i + threadIdx.x];
        col_ind_tile[i + threadIdx.x] =
            gA.col_ind[re - __anf0 + i + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk, gB + (cols * kr + n_idx + __anf02 * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 4U), out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x4x1_on(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC,
                                    cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x4x1_on_0,
              rows * (cols / 4U + (uint32_t) (cols % 4U != 0U)), 1U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x8x2_on
*/
static void
__hoisted_g_spmm_f32_32x8x2_on_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 8U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++)
            vec_memcpy(elems_tile + (i0 * 2U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 2U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 2U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 2U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 2U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 2U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 2U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 4U; i++)
                vec_memcpy(elems_tile + (i * 2U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 2U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 2U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 2U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 2U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 2U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 1U - threadIdx.x) / 2U; i++) {
        elems_tile[i * 2U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 2U + threadIdx.x];
        col_ind_tile[i * 2U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 2U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 2U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 2U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 2U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 2U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x8x2_on(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC,
                                    cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x8x2_on_0,
              rows * (cols / 8U + (uint32_t) (cols % 8U != 0U)), 2U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x16x4_on
*/
static void
__hoisted_g_spmm_f32_32x16x4_on_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 16U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 2U; i0++)
            vec_memcpy(elems_tile + (i0 * 4U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 4U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 4U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 4U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 4U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 4U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 2U; i++)
                vec_memcpy(elems_tile + (i * 4U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 4U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 4U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 4U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 4U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 4U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 3U - threadIdx.x) / 4U; i++) {
        elems_tile[i * 4U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 4U + threadIdx.x];
        col_ind_tile[i * 4U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 4U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 4U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 4U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 4U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 4U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x16x4_on(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x16x4_on_0,
              rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)), 4U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x32x8_on
*/
static void
__hoisted_g_spmm_f32_32x32x8_on_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 8U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 8U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 8U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 8U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 8U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 8U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 8U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 8U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 8U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 8U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 7U - threadIdx.x) / 8U; i++) {
        elems_tile[i * 8U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 8U + threadIdx.x];
        col_ind_tile[i * 8U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 8U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 8U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 8U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x32x8_on(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x32x8_on_0,
              rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 8U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_32x64x8_on
*/
static void
__hoisted_g_spmm_f32_32x64x8_on_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile = (uint32_t *) KPR_SHMEM_AT(128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i0 = 0U;
        for (; i0 < 1U; i0++)
            vec_memcpy(elems_tile + (i0 * 8U + threadIdx.x) * 4U,
                       gA.elems + (ri_ + (i0 * 8U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 8U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 8U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 32U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 8U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i = 0U;
            for (; i < 1U; i++)
                vec_memcpy(elems_tile + (i * 8U + threadIdx.x) * 4U,
                           gA.elems + (off + (i * 8U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 8U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 8U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 8U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
    for (; i < (__anf0 + 7U - threadIdx.x) / 8U; i++) {
        elems_tile[i * 8U + threadIdx.x] =
            gA.elems[re - __anf0 + i * 8U + threadIdx.x];
        col_ind_tile[i * 8U + threadIdx.x] =
            gA.col_ind[re - __anf0 + i * 8U + threadIdx.x];
    }
    __syncthreads();
    uint32_t k = 0U;
    for (; k < __anf0; k++) {
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 8U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 8U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_32x64x8_on(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(256U);
    KPR_KCALL(__hoisted_g_spmm_f32_32x64x8_on_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 8U, 256U, s,
              rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x16
*/
static void
__hoisted_g_spmm_f32_64x64x16_0(uint32_t rows, uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x64x16(uint32_t rows, uint32_t shared,
                                   uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 512U, s,
              rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x64x16_on
*/
static void
__hoisted_g_spmm_f32_64x64x16_on_0(uint32_t rows, uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x64x16_on(uint32_t rows, uint32_t shared,
                                      uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__float gA,
                                      uint32_t *row_indices, float *gB,
                                      float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x16_on_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 512U, s,
              rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x16
*/
static void
__hoisted_g_spmm_f32_64x128x16_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x128x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 512U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x128x16_on
*/
static void
__hoisted_g_spmm_f32_64x128x16_on_0(uint32_t rows, uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x128x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x16_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 512U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x16
*/
static void
__hoisted_g_spmm_f32_64x256x16_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x256x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 512U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x256x16_on
*/
static void
__hoisted_g_spmm_f32_64x256x16_on_0(uint32_t rows, uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x256x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x16_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 512U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x16
*/
static void
__hoisted_g_spmm_f32_64x512x16_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x512x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 512U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_64x512x16_on
*/
static void
__hoisted_g_spmm_f32_64x512x16_on_0(uint32_t rows, uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 64U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_64x512x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(512U);
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x16_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 512U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x16
*/
static void
__hoisted_g_spmm_f32_128x64x16_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x64x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x64x16_on
*/
static void
__hoisted_g_spmm_f32_128x64x16_on_0(uint32_t rows, uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x64x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x16_on_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x16
*/
static void
__hoisted_g_spmm_f32_128x128x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x128x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x16_on
*/
static void
__hoisted_g_spmm_f32_128x128x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x128x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x16_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x32
*/
static void
__hoisted_g_spmm_f32_128x128x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x128x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x128x32_on
*/
static void
__hoisted_g_spmm_f32_128x128x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x128x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x32_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x16
*/
static void
__hoisted_g_spmm_f32_128x256x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x256x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x16_on
*/
static void
__hoisted_g_spmm_f32_128x256x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x256x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x16_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x32
*/
static void
__hoisted_g_spmm_f32_128x256x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x256x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x256x32_on
*/
static void
__hoisted_g_spmm_f32_128x256x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x256x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x32_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x16
*/
static void
__hoisted_g_spmm_f32_128x512x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x512x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x16_on
*/
static void
__hoisted_g_spmm_f32_128x512x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x512x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x16_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x32
*/
static void
__hoisted_g_spmm_f32_128x512x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x512x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_128x512x32_on
*/
static void
__hoisted_g_spmm_f32_128x512x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 128U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_128x512x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x32_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U, 1024U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x16
*/
static void
__hoisted_g_spmm_f32_256x64x16_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x64x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x64x16_on
*/
static void
__hoisted_g_spmm_f32_256x64x16_on_0(uint32_t rows, uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x64x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x16_on_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x16
*/
static void
__hoisted_g_spmm_f32_256x128x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x128x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x16_on
*/
static void
__hoisted_g_spmm_f32_256x128x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x128x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x16_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x32
*/
static void
__hoisted_g_spmm_f32_256x128x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x128x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x128x32_on
*/
static void
__hoisted_g_spmm_f32_256x128x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x128x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x32_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x16
*/
static void
__hoisted_g_spmm_f32_256x256x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x256x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x16_on
*/
static void
__hoisted_g_spmm_f32_256x256x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x256x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x16_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x32
*/
static void
__hoisted_g_spmm_f32_256x256x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x256x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x32_on
*/
static void
__hoisted_g_spmm_f32_256x256x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x256x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x32_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x64
*/
static void
__hoisted_g_spmm_f32_256x256x64_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x256x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x64_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x256x64_on
*/
static void
__hoisted_g_spmm_f32_256x256x64_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x256x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x64_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x16
*/
static void
__hoisted_g_spmm_f32_256x512x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x512x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x16_on
*/
static void
__hoisted_g_spmm_f32_256x512x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x512x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x16_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x32
*/
static void
__hoisted_g_spmm_f32_256x512x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x512x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x32_on
*/
static void
__hoisted_g_spmm_f32_256x512x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x512x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x32_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x64
*/
static void
__hoisted_g_spmm_f32_256x512x64_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x512x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x64_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_256x512x64_on
*/
static void
__hoisted_g_spmm_f32_256x512x64_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 256U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_256x512x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x64_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U, 2048U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x16
*/
static void
__hoisted_g_spmm_f32_512x64x16_0(uint32_t rows, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x64x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x16_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x64x16_on
*/
static void
__hoisted_g_spmm_f32_512x64x16_on_0(uint32_t rows, uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x64x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x16_on_0,
              rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x16
*/
static void
__hoisted_g_spmm_f32_512x128x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x128x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x16_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x16_on
*/
static void
__hoisted_g_spmm_f32_512x128x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x128x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x16_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x32
*/
static void
__hoisted_g_spmm_f32_512x128x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x128x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x32_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x128x32_on
*/
static void
__hoisted_g_spmm_f32_512x128x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x128x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x32_on_0,
              rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x16
*/
static void
__hoisted_g_spmm_f32_512x256x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x256x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x16_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x16_on
*/
static void
__hoisted_g_spmm_f32_512x256x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x256x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x16_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x32
*/
static void
__hoisted_g_spmm_f32_512x256x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x256x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x32_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x32_on
*/
static void
__hoisted_g_spmm_f32_512x256x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x256x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x32_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x64
*/
static void
__hoisted_g_spmm_f32_512x256x64_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x256x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x64_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x256x64_on
*/
static void
__hoisted_g_spmm_f32_512x256x64_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x256x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x64_on_0,
              rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x16
*/
static void
__hoisted_g_spmm_f32_512x512x16_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x16_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x16_on
*/
static void
__hoisted_g_spmm_f32_512x512x16_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 16U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 16U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 16U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 16U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 16U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 16U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 8U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 16U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 8U; i0++)
        if (n_idx + i0 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 16U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x16_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x32
*/
static void
__hoisted_g_spmm_f32_512x512x32_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x32_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x32_on
*/
static void
__hoisted_g_spmm_f32_512x512x32_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 32U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 32U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 32U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 32U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 32U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 32U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 4U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 32U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 4U; i0++)
        if (n_idx + i0 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 32U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x32_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x64
*/
static void
__hoisted_g_spmm_f32_512x512x64_0(uint32_t rows, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x64_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x64_on
*/
static void
__hoisted_g_spmm_f32_512x512x64_on_0(uint32_t rows, uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 64U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 2U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 64U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 64U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 64U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 64U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 64U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 2U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 64U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 2U; i0++)
        if (n_idx + i0 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 64U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x64_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U, 4096U,
              s, rows, cols, gA, row_indices, gB, gC);
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x128
*/
static void
__hoisted_g_spmm_f32_512x512x128_0(uint32_t rows, uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 128U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 128U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 128U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 128U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 128U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 128U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 128U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 128U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 128U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 128U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 128U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 128U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 128U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 128U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 128U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x128(uint32_t rows, uint32_t shared,
                                      uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__float gA,
                                      uint32_t *row_indices, float *gB,
                                      float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x128_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 128U,
              4096U, s, rows, cols, gA, row_indices, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_spmm_f32_512x512x128_on
*/
static void
__hoisted_g_spmm_f32_512x512x128_on_0(uint32_t rows, uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__float gA,
                                      uint32_t *row_indices, float *gB,
                                      float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile = (float *) KPR_SHMEM_AT(0U);
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
                       gA.elems + (ri_ + (i0 * 128U + threadIdx.x) * 4U));
        uint32_t i1 = 0U;
        for (; i1 < 1U; i1++)
            vec_memcpy(col_ind_tile + (i1 * 128U + threadIdx.x) * 4U,
                       gA.col_ind + (ri_ + (i1 * 128U + threadIdx.x) * 4U));
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i2 = 0U;
        for (; i2 < to_; i2++)
            elems_tile[i2 * 128U + threadIdx.x] = 0.0f;
        __syncthreads();
        uint32_t k = 0U;
        for (; k < 512U; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile[kv];
            float kx = elems_tile[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 128U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                               gB + (cols * kr + n_idx + __anf02 * 128U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
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
                           gA.elems + (off + (i * 128U + threadIdx.x) * 4U));
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0++)
                vec_memcpy(col_ind_tile + (i0 * 128U + threadIdx.x) * 4U,
                           gA.col_ind + (off + (i0 * 128U + threadIdx.x) * 4U));
            __syncthreads();
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile[kv];
                float kx = elems_tile[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf13 = k1;
                    uint32_t __anf012 = k1;
                    if (n_idx + __anf012 * 128U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk, gB + (cols * kr + n_idx +
                                                 __anf012 * 128U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf13 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
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
        uint32_t kv = k;
        uint32_t kr = col_ind_tile[kv];
        float kx = elems_tile[kv];
        uint32_t k1 = 0U;
        for (; k1 < 1U; k1++) {
            uint32_t __anf11 = k1;
            uint32_t __anf02 = k1;
            if (n_idx + __anf02 * 128U * 4U < cols) {
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(lchunk,
                           gB + (cols * kr + n_idx + __anf02 * 128U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf11 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i0 = 0U;
    for (; i0 < 1U; i0++)
        if (n_idx + i0 * 128U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + i0 * 128U * 4U),
                       out + i0 * 4U);
}

void Klas_SPMM_g_spmm_f32_512x512x128_on(uint32_t rows, uint32_t shared,
                                         uint32_t cols,
                                         Kuiper_Sparse_Matrix_smatrix__float gA,
                                         uint32_t *row_indices, float *gB,
                                         float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x128_on_0,
              rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 128U,
              4096U, s, rows, cols, gA, row_indices, gB, gC);
}

void Klas_SPMM_spmm_f32_dispatch(uint32_t rows, uint32_t shared, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC)
{
    if (cols % 64U == 0U)
        Klas_SPMM_g_spmm_f32_32x64x8(rows, shared, cols, gA, row_indices, gB,
                                     gC);
    else if (cols % 32U == 0U)
        Klas_SPMM_g_spmm_f32_32x32x8(rows, shared, cols, gA, row_indices, gB,
                                     gC);
    else if (cols % 16U == 0U)
        Klas_SPMM_g_spmm_f32_32x16x4(rows, shared, cols, gA, row_indices, gB,
                                     gC);
    else if (cols % 8U == 0U)
        Klas_SPMM_g_spmm_f32_32x8x2(rows, shared, cols, gA, row_indices, gB,
                                    gC);
    else
        Klas_SPMM_g_spmm_f32_32x4x1(rows, shared, cols, gA, row_indices, gB,
                                    gC);
}
