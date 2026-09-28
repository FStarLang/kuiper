
#include "Klas_SPMM.h"

__global__ __launch_bounds__(32)
/**
  hoisted when extracting spmm_u32
*/
static void
__hoisted_spmm_u32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__uint32_t gA, uint32_t cols, uint32_t *gB,
    uint32_t *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    uint32_t *elems_tile0 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    uint32_t out[4U] = {0U};
    if (nnz >= 128U) {
        uint32_t i_kpr0 = 0U;
        for (; i_kpr0 < 1U; i_kpr0++) {
            uint32_t j_kpr1 = i_kpr0;
            vec_memcpy(elems_tile0 + (j_kpr1 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr2 = 0U;
        for (; i_kpr2 < 1U; i_kpr2++) {
            uint32_t j_kpr3 = i_kpr2;
            vec_memcpy(col_ind_tile0 + (j_kpr3 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr3 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr4 = 0U;
        for (; i_kpr4 < to_; i_kpr4++)
            elems_tile0[i_kpr4 * 32U + threadIdx.x] = 0U;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                uint32_t kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    uint32_t lchunk[4U] = {0U};
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr6 = 0U;
            for (; i_kpr6 < 1U; i_kpr6++) {
                uint32_t j_kpr7 = i_kpr6;
                vec_memcpy(elems_tile0 + (j_kpr7 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr7 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr8 = 0U;
            for (; i_kpr8 < 1U; i_kpr8++) {
                uint32_t j_kpr9 = i_kpr8;
                vec_memcpy(col_ind_tile0 + (j_kpr9 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr9 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    uint32_t kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        uint32_t lchunk[4U] = {0U};
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr11 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr10 = 0U;
    for (; i_kpr10 < hi_kpr11; i_kpr10++) {
        uint32_t j_kpr12 = i_kpr10;
        elems_tile0[j_kpr12 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr12 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr12 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr12 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            uint32_t kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                uint32_t lchunk[4U] = {0U};
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr13 = 0U;
    for (; i_kpr13 < 1U; i_kpr13++) {
        uint32_t j_kpr14 = i_kpr13;
        if (n_idx + j_kpr14 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr14 * 32U * 4U),
                out + j_kpr14 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting spmm_f32
*/
static void
__hoisted_spmm_f32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr15 = 0U;
        for (; i_kpr15 < 2U; i_kpr15++) {
            uint32_t j_kpr16 = i_kpr15;
            vec_memcpy(elems_tile0 + (j_kpr16 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr16 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr17 = 0U;
        for (; i_kpr17 < 2U; i_kpr17++) {
            uint32_t j_kpr18 = i_kpr17;
            vec_memcpy(col_ind_tile0 + (j_kpr18 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr18 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr19 = 0U;
        for (; i_kpr19 < to_; i_kpr19++)
            elems_tile0[i_kpr19 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr21 = 0U;
            for (; i_kpr21 < 2U; i_kpr21++) {
                uint32_t j_kpr22 = i_kpr21;
                vec_memcpy(elems_tile0 + (j_kpr22 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr22 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr23 = 0U;
            for (; i_kpr23 < 2U; i_kpr23++) {
                uint32_t j_kpr24 = i_kpr23;
                vec_memcpy(col_ind_tile0 + (j_kpr24 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr24 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 64U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr26 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr25 = 0U;
    for (; i_kpr25 < hi_kpr26; i_kpr25++) {
        uint32_t j_kpr27 = i_kpr25;
        elems_tile0[j_kpr27 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr27 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr27 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr27 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr28 = 0U;
    for (; i_kpr28 < 2U; i_kpr28++) {
        uint32_t j_kpr29 = i_kpr28;
        if (n_idx + j_kpr29 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr29 * 64U * 4U),
                out + j_kpr29 * 4U);
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting g_spmm_f32_32x4x1
*/
static void
__hoisted_g_spmm_f32_32x4x1_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 4U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr30 = 0U;
        for (; i_kpr30 < 8U; i_kpr30++) {
            uint32_t j_kpr31 = i_kpr30;
            vec_memcpy(elems_tile0 + (j_kpr31 + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr31 + threadIdx.x) * 4U));
        }
        uint32_t i_kpr32 = 0U;
        for (; i_kpr32 < 8U; i_kpr32++) {
            uint32_t j_kpr33 = i_kpr32;
            vec_memcpy(col_ind_tile0 + (j_kpr33 + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr33 + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i_kpr34 = 0U;
        for (; i_kpr34 < to_; i_kpr34++)
            elems_tile0[i_kpr34 + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk, gB + (cols * kr + n_idx + __anf01 * 4U));
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
            uint32_t i_kpr36 = 0U;
            for (; i_kpr36 < 8U; i_kpr36++) {
                uint32_t j_kpr37 = i_kpr36;
                vec_memcpy(elems_tile0 + (j_kpr37 + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr37 + threadIdx.x) * 4U));
            }
            uint32_t i_kpr38 = 0U;
            for (; i_kpr38 < 8U; i_kpr38++) {
                uint32_t j_kpr39 = i_kpr38;
                vec_memcpy(col_ind_tile0 + (j_kpr39 + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr39 + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(
                            lchunk, gB + (cols * kr + n_idx + __anf03 * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr41 = __anf01 - threadIdx.x;
    uint32_t i_kpr40 = 0U;
    for (; i_kpr40 < hi_kpr41; i_kpr40++) {
        uint32_t j_kpr42 = i_kpr40;
        elems_tile0[j_kpr42 + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr42 + threadIdx.x];
        col_ind_tile0[j_kpr42 + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr42 + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
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
    uint32_t i_kpr43 = 0U;
    for (; i_kpr43 < 1U; i_kpr43++) {
        uint32_t j_kpr44 = i_kpr43;
        if (n_idx + j_kpr44 * 4U < cols)
            vec_memcpy(
                gC + (cols * m_idx + n_idx + j_kpr44 * 4U), out + j_kpr44 * 4U);
    }
}

__global__ __launch_bounds__(2)
/**
  hoisted when extracting g_spmm_f32_32x8x2
*/
static void
__hoisted_g_spmm_f32_32x8x2_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 8U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr45 = 0U;
        for (; i_kpr45 < 4U; i_kpr45++) {
            uint32_t j_kpr46 = i_kpr45;
            vec_memcpy(elems_tile0 + (j_kpr46 * 2U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr46 * 2U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr47 = 0U;
        for (; i_kpr47 < 4U; i_kpr47++) {
            uint32_t j_kpr48 = i_kpr47;
            vec_memcpy(col_ind_tile0 + (j_kpr48 * 2U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr48 * 2U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i_kpr49 = 0U;
        for (; i_kpr49 < to_; i_kpr49++)
            elems_tile0[i_kpr49 * 2U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 2U * 4U));
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
            uint32_t i_kpr51 = 0U;
            for (; i_kpr51 < 4U; i_kpr51++) {
                uint32_t j_kpr52 = i_kpr51;
                vec_memcpy(elems_tile0 + (j_kpr52 * 2U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr52 * 2U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr53 = 0U;
            for (; i_kpr53 < 4U; i_kpr53++) {
                uint32_t j_kpr54 = i_kpr53;
                vec_memcpy(col_ind_tile0 + (j_kpr54 * 2U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr54 * 2U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 2U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr56 = (__anf01 + 1U - threadIdx.x) / 2U;
    uint32_t i_kpr55 = 0U;
    for (; i_kpr55 < hi_kpr56; i_kpr55++) {
        uint32_t j_kpr57 = i_kpr55;
        elems_tile0[j_kpr57 * 2U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr57 * 2U + threadIdx.x];
        col_ind_tile0[j_kpr57 * 2U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr57 * 2U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 2U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr58 = 0U;
    for (; i_kpr58 < 1U; i_kpr58++) {
        uint32_t j_kpr59 = i_kpr58;
        if (n_idx + j_kpr59 * 2U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr59 * 2U * 4U),
                out + j_kpr59 * 4U);
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f32_32x16x4
*/
static void
__hoisted_g_spmm_f32_32x16x4_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 16U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr60 = 0U;
        for (; i_kpr60 < 2U; i_kpr60++) {
            uint32_t j_kpr61 = i_kpr60;
            vec_memcpy(elems_tile0 + (j_kpr61 * 4U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr61 * 4U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr62 = 0U;
        for (; i_kpr62 < 2U; i_kpr62++) {
            uint32_t j_kpr63 = i_kpr62;
            vec_memcpy(col_ind_tile0 + (j_kpr63 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr63 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr64 = 0U;
        for (; i_kpr64 < to_; i_kpr64++)
            elems_tile0[i_kpr64 * 4U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 4U * 4U));
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
            uint32_t i_kpr66 = 0U;
            for (; i_kpr66 < 2U; i_kpr66++) {
                uint32_t j_kpr67 = i_kpr66;
                vec_memcpy(elems_tile0 + (j_kpr67 * 4U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr67 * 4U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr68 = 0U;
            for (; i_kpr68 < 2U; i_kpr68++) {
                uint32_t j_kpr69 = i_kpr68;
                vec_memcpy(col_ind_tile0 + (j_kpr69 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr69 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 4U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr71 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr70 = 0U;
    for (; i_kpr70 < hi_kpr71; i_kpr70++) {
        uint32_t j_kpr72 = i_kpr70;
        elems_tile0[j_kpr72 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr72 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr72 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr72 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 4U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr73 = 0U;
    for (; i_kpr73 < 1U; i_kpr73++) {
        uint32_t j_kpr74 = i_kpr73;
        if (n_idx + j_kpr74 * 4U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr74 * 4U * 4U),
                out + j_kpr74 * 4U);
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x32x8
*/
static void
__hoisted_g_spmm_f32_32x32x8_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr75 = 0U;
        for (; i_kpr75 < 1U; i_kpr75++) {
            uint32_t j_kpr76 = i_kpr75;
            vec_memcpy(elems_tile0 + (j_kpr76 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr76 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr77 = 0U;
        for (; i_kpr77 < 1U; i_kpr77++) {
            uint32_t j_kpr78 = i_kpr77;
            vec_memcpy(col_ind_tile0 + (j_kpr78 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr78 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr79 = 0U;
        for (; i_kpr79 < to_; i_kpr79++)
            elems_tile0[i_kpr79 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 8U * 4U));
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
            uint32_t i_kpr81 = 0U;
            for (; i_kpr81 < 1U; i_kpr81++) {
                uint32_t j_kpr82 = i_kpr81;
                vec_memcpy(elems_tile0 + (j_kpr82 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr82 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr83 = 0U;
            for (; i_kpr83 < 1U; i_kpr83++) {
                uint32_t j_kpr84 = i_kpr83;
                vec_memcpy(col_ind_tile0 + (j_kpr84 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr84 * 8U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr86 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr85 = 0U;
    for (; i_kpr85 < hi_kpr86; i_kpr85++) {
        uint32_t j_kpr87 = i_kpr85;
        elems_tile0[j_kpr87 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr87 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr87 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr87 * 8U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr88 = 0U;
    for (; i_kpr88 < 1U; i_kpr88++) {
        uint32_t j_kpr89 = i_kpr88;
        if (n_idx + j_kpr89 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr89 * 8U * 4U),
                out + j_kpr89 * 4U);
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x64x8
*/
static void
__hoisted_g_spmm_f32_32x64x8_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr90 = 0U;
        for (; i_kpr90 < 1U; i_kpr90++) {
            uint32_t j_kpr91 = i_kpr90;
            vec_memcpy(elems_tile0 + (j_kpr91 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr91 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr92 = 0U;
        for (; i_kpr92 < 1U; i_kpr92++) {
            uint32_t j_kpr93 = i_kpr92;
            vec_memcpy(col_ind_tile0 + (j_kpr93 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr93 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr94 = 0U;
        for (; i_kpr94 < to_; i_kpr94++)
            elems_tile0[i_kpr94 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 8U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr96 = 0U;
            for (; i_kpr96 < 1U; i_kpr96++) {
                uint32_t j_kpr97 = i_kpr96;
                vec_memcpy(elems_tile0 + (j_kpr97 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr97 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr98 = 0U;
            for (; i_kpr98 < 1U; i_kpr98++) {
                uint32_t j_kpr99 = i_kpr98;
                vec_memcpy(col_ind_tile0 + (j_kpr99 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr99 * 8U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 8U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 8U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr101 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr100 = 0U;
    for (; i_kpr100 < hi_kpr101; i_kpr100++) {
        uint32_t j_kpr102 = i_kpr100;
        elems_tile0[j_kpr102 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr102 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr102 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr102 * 8U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 8U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr103 = 0U;
    for (; i_kpr103 < 2U; i_kpr103++) {
        uint32_t j_kpr104 = i_kpr103;
        if (n_idx + j_kpr104 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr104 * 8U * 4U),
                out + j_kpr104 * 4U);
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting g_spmm_f32_32x4x1_on
*/
static void
__hoisted_g_spmm_f32_32x4x1_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 4U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr105 = 0U;
        for (; i_kpr105 < 8U; i_kpr105++) {
            uint32_t j_kpr106 = i_kpr105;
            vec_memcpy(elems_tile0 + (j_kpr106 + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr106 + threadIdx.x) * 4U));
        }
        uint32_t i_kpr107 = 0U;
        for (; i_kpr107 < 8U; i_kpr107++) {
            uint32_t j_kpr108 = i_kpr107;
            vec_memcpy(col_ind_tile0 + (j_kpr108 + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr108 + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i_kpr109 = 0U;
        for (; i_kpr109 < to_; i_kpr109++)
            elems_tile0[i_kpr109 + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk, gB + (cols * kr + n_idx + __anf01 * 4U));
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
            uint32_t i_kpr111 = 0U;
            for (; i_kpr111 < 8U; i_kpr111++) {
                uint32_t j_kpr112 = i_kpr111;
                vec_memcpy(elems_tile0 + (j_kpr112 + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr112 + threadIdx.x) * 4U));
            }
            uint32_t i_kpr113 = 0U;
            for (; i_kpr113 < 8U; i_kpr113++) {
                uint32_t j_kpr114 = i_kpr113;
                vec_memcpy(col_ind_tile0 + (j_kpr114 + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr114 + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(
                            lchunk, gB + (cols * kr + n_idx + __anf03 * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr116 = __anf01 - threadIdx.x;
    uint32_t i_kpr115 = 0U;
    for (; i_kpr115 < hi_kpr116; i_kpr115++) {
        uint32_t j_kpr117 = i_kpr115;
        elems_tile0[j_kpr117 + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr117 + threadIdx.x];
        col_ind_tile0[j_kpr117 + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr117 + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
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
    uint32_t i_kpr118 = 0U;
    for (; i_kpr118 < 1U; i_kpr118++) {
        uint32_t j_kpr119 = i_kpr118;
        if (n_idx + j_kpr119 * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr119 * 4U),
                out + j_kpr119 * 4U);
    }
}

__global__ __launch_bounds__(2)
/**
  hoisted when extracting g_spmm_f32_32x8x2_on
*/
static void
__hoisted_g_spmm_f32_32x8x2_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 8U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr120 = 0U;
        for (; i_kpr120 < 4U; i_kpr120++) {
            uint32_t j_kpr121 = i_kpr120;
            vec_memcpy(elems_tile0 + (j_kpr121 * 2U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr121 * 2U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr122 = 0U;
        for (; i_kpr122 < 4U; i_kpr122++) {
            uint32_t j_kpr123 = i_kpr122;
            vec_memcpy(col_ind_tile0 + (j_kpr123 * 2U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr123 * 2U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i_kpr124 = 0U;
        for (; i_kpr124 < to_; i_kpr124++)
            elems_tile0[i_kpr124 * 2U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 2U * 4U));
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
            uint32_t i_kpr126 = 0U;
            for (; i_kpr126 < 4U; i_kpr126++) {
                uint32_t j_kpr127 = i_kpr126;
                vec_memcpy(elems_tile0 + (j_kpr127 * 2U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr127 * 2U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr128 = 0U;
            for (; i_kpr128 < 4U; i_kpr128++) {
                uint32_t j_kpr129 = i_kpr128;
                vec_memcpy(col_ind_tile0 + (j_kpr129 * 2U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr129 * 2U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 2U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr131 = (__anf01 + 1U - threadIdx.x) / 2U;
    uint32_t i_kpr130 = 0U;
    for (; i_kpr130 < hi_kpr131; i_kpr130++) {
        uint32_t j_kpr132 = i_kpr130;
        elems_tile0[j_kpr132 * 2U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr132 * 2U + threadIdx.x];
        col_ind_tile0[j_kpr132 * 2U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr132 * 2U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 2U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr133 = 0U;
    for (; i_kpr133 < 1U; i_kpr133++) {
        uint32_t j_kpr134 = i_kpr133;
        if (n_idx + j_kpr134 * 2U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr134 * 2U * 4U),
                out + j_kpr134 * 4U);
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f32_32x16x4_on
*/
static void
__hoisted_g_spmm_f32_32x16x4_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 16U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr135 = 0U;
        for (; i_kpr135 < 2U; i_kpr135++) {
            uint32_t j_kpr136 = i_kpr135;
            vec_memcpy(elems_tile0 + (j_kpr136 * 4U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr136 * 4U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr137 = 0U;
        for (; i_kpr137 < 2U; i_kpr137++) {
            uint32_t j_kpr138 = i_kpr137;
            vec_memcpy(col_ind_tile0 + (j_kpr138 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr138 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr139 = 0U;
        for (; i_kpr139 < to_; i_kpr139++)
            elems_tile0[i_kpr139 * 4U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 4U * 4U));
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
            uint32_t i_kpr141 = 0U;
            for (; i_kpr141 < 2U; i_kpr141++) {
                uint32_t j_kpr142 = i_kpr141;
                vec_memcpy(elems_tile0 + (j_kpr142 * 4U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr142 * 4U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr143 = 0U;
            for (; i_kpr143 < 2U; i_kpr143++) {
                uint32_t j_kpr144 = i_kpr143;
                vec_memcpy(col_ind_tile0 + (j_kpr144 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr144 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 4U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr146 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr145 = 0U;
    for (; i_kpr145 < hi_kpr146; i_kpr145++) {
        uint32_t j_kpr147 = i_kpr145;
        elems_tile0[j_kpr147 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr147 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr147 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr147 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 4U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr148 = 0U;
    for (; i_kpr148 < 1U; i_kpr148++) {
        uint32_t j_kpr149 = i_kpr148;
        if (n_idx + j_kpr149 * 4U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr149 * 4U * 4U),
                out + j_kpr149 * 4U);
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x32x8_on
*/
static void
__hoisted_g_spmm_f32_32x32x8_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr150 = 0U;
        for (; i_kpr150 < 1U; i_kpr150++) {
            uint32_t j_kpr151 = i_kpr150;
            vec_memcpy(elems_tile0 + (j_kpr151 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr151 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr152 = 0U;
        for (; i_kpr152 < 1U; i_kpr152++) {
            uint32_t j_kpr153 = i_kpr152;
            vec_memcpy(col_ind_tile0 + (j_kpr153 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr153 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr154 = 0U;
        for (; i_kpr154 < to_; i_kpr154++)
            elems_tile0[i_kpr154 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 8U * 4U));
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
            uint32_t i_kpr156 = 0U;
            for (; i_kpr156 < 1U; i_kpr156++) {
                uint32_t j_kpr157 = i_kpr156;
                vec_memcpy(elems_tile0 + (j_kpr157 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr157 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr158 = 0U;
            for (; i_kpr158 < 1U; i_kpr158++) {
                uint32_t j_kpr159 = i_kpr158;
                vec_memcpy(col_ind_tile0 + (j_kpr159 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr159 * 8U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr161 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr160 = 0U;
    for (; i_kpr160 < hi_kpr161; i_kpr160++) {
        uint32_t j_kpr162 = i_kpr160;
        elems_tile0[j_kpr162 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr162 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr162 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr162 * 8U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr163 = 0U;
    for (; i_kpr163 < 1U; i_kpr163++) {
        uint32_t j_kpr164 = i_kpr163;
        if (n_idx + j_kpr164 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr164 * 8U * 4U),
                out + j_kpr164 * 4U);
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x64x8_on
*/
static void
__hoisted_g_spmm_f32_32x64x8_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr165 = 0U;
        for (; i_kpr165 < 1U; i_kpr165++) {
            uint32_t j_kpr166 = i_kpr165;
            vec_memcpy(elems_tile0 + (j_kpr166 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr166 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr167 = 0U;
        for (; i_kpr167 < 1U; i_kpr167++) {
            uint32_t j_kpr168 = i_kpr167;
            vec_memcpy(col_ind_tile0 + (j_kpr168 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr168 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr169 = 0U;
        for (; i_kpr169 < to_; i_kpr169++)
            elems_tile0[i_kpr169 * 8U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 8U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 8U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr171 = 0U;
            for (; i_kpr171 < 1U; i_kpr171++) {
                uint32_t j_kpr172 = i_kpr171;
                vec_memcpy(elems_tile0 + (j_kpr172 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr172 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr173 = 0U;
            for (; i_kpr173 < 1U; i_kpr173++) {
                uint32_t j_kpr174 = i_kpr173;
                vec_memcpy(col_ind_tile0 + (j_kpr174 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr174 * 8U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 8U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 8U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr176 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr175 = 0U;
    for (; i_kpr175 < hi_kpr176; i_kpr175++) {
        uint32_t j_kpr177 = i_kpr175;
        elems_tile0[j_kpr177 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr177 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr177 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr177 * 8U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 8U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 8U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr178 = 0U;
    for (; i_kpr178 < 2U; i_kpr178++) {
        uint32_t j_kpr179 = i_kpr178;
        if (n_idx + j_kpr179 * 8U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr179 * 8U * 4U),
                out + j_kpr179 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x64x16
*/
static void
__hoisted_g_spmm_f32_64x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr180 = 0U;
        for (; i_kpr180 < 1U; i_kpr180++) {
            uint32_t j_kpr181 = i_kpr180;
            vec_memcpy(elems_tile0 + (j_kpr181 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr181 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr182 = 0U;
        for (; i_kpr182 < 1U; i_kpr182++) {
            uint32_t j_kpr183 = i_kpr182;
            vec_memcpy(col_ind_tile0 + (j_kpr183 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr183 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr184 = 0U;
        for (; i_kpr184 < to_; i_kpr184++)
            elems_tile0[i_kpr184 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr186 = 0U;
            for (; i_kpr186 < 1U; i_kpr186++) {
                uint32_t j_kpr187 = i_kpr186;
                vec_memcpy(elems_tile0 + (j_kpr187 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr187 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr188 = 0U;
            for (; i_kpr188 < 1U; i_kpr188++) {
                uint32_t j_kpr189 = i_kpr188;
                vec_memcpy(col_ind_tile0 + (j_kpr189 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr189 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr191 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr190 = 0U;
    for (; i_kpr190 < hi_kpr191; i_kpr190++) {
        uint32_t j_kpr192 = i_kpr190;
        elems_tile0[j_kpr192 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr192 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr192 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr192 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr193 = 0U;
    for (; i_kpr193 < 1U; i_kpr193++) {
        uint32_t j_kpr194 = i_kpr193;
        if (n_idx + j_kpr194 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr194 * 16U * 4U),
                out + j_kpr194 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x64x16_on
*/
static void
__hoisted_g_spmm_f32_64x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr195 = 0U;
        for (; i_kpr195 < 1U; i_kpr195++) {
            uint32_t j_kpr196 = i_kpr195;
            vec_memcpy(elems_tile0 + (j_kpr196 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr196 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr197 = 0U;
        for (; i_kpr197 < 1U; i_kpr197++) {
            uint32_t j_kpr198 = i_kpr197;
            vec_memcpy(col_ind_tile0 + (j_kpr198 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr198 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr199 = 0U;
        for (; i_kpr199 < to_; i_kpr199++)
            elems_tile0[i_kpr199 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr201 = 0U;
            for (; i_kpr201 < 1U; i_kpr201++) {
                uint32_t j_kpr202 = i_kpr201;
                vec_memcpy(elems_tile0 + (j_kpr202 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr202 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr203 = 0U;
            for (; i_kpr203 < 1U; i_kpr203++) {
                uint32_t j_kpr204 = i_kpr203;
                vec_memcpy(col_ind_tile0 + (j_kpr204 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr204 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr206 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr205 = 0U;
    for (; i_kpr205 < hi_kpr206; i_kpr205++) {
        uint32_t j_kpr207 = i_kpr205;
        elems_tile0[j_kpr207 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr207 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr207 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr207 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr208 = 0U;
    for (; i_kpr208 < 1U; i_kpr208++) {
        uint32_t j_kpr209 = i_kpr208;
        if (n_idx + j_kpr209 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr209 * 16U * 4U),
                out + j_kpr209 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x128x16
*/
static void
__hoisted_g_spmm_f32_64x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr210 = 0U;
        for (; i_kpr210 < 1U; i_kpr210++) {
            uint32_t j_kpr211 = i_kpr210;
            vec_memcpy(elems_tile0 + (j_kpr211 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr211 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr212 = 0U;
        for (; i_kpr212 < 1U; i_kpr212++) {
            uint32_t j_kpr213 = i_kpr212;
            vec_memcpy(col_ind_tile0 + (j_kpr213 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr213 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr214 = 0U;
        for (; i_kpr214 < to_; i_kpr214++)
            elems_tile0[i_kpr214 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr216 = 0U;
            for (; i_kpr216 < 1U; i_kpr216++) {
                uint32_t j_kpr217 = i_kpr216;
                vec_memcpy(elems_tile0 + (j_kpr217 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr217 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr218 = 0U;
            for (; i_kpr218 < 1U; i_kpr218++) {
                uint32_t j_kpr219 = i_kpr218;
                vec_memcpy(col_ind_tile0 + (j_kpr219 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr219 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr221 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr220 = 0U;
    for (; i_kpr220 < hi_kpr221; i_kpr220++) {
        uint32_t j_kpr222 = i_kpr220;
        elems_tile0[j_kpr222 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr222 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr222 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr222 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr223 = 0U;
    for (; i_kpr223 < 2U; i_kpr223++) {
        uint32_t j_kpr224 = i_kpr223;
        if (n_idx + j_kpr224 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr224 * 16U * 4U),
                out + j_kpr224 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x128x16_on
*/
static void
__hoisted_g_spmm_f32_64x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr225 = 0U;
        for (; i_kpr225 < 1U; i_kpr225++) {
            uint32_t j_kpr226 = i_kpr225;
            vec_memcpy(elems_tile0 + (j_kpr226 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr226 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr227 = 0U;
        for (; i_kpr227 < 1U; i_kpr227++) {
            uint32_t j_kpr228 = i_kpr227;
            vec_memcpy(col_ind_tile0 + (j_kpr228 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr228 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr229 = 0U;
        for (; i_kpr229 < to_; i_kpr229++)
            elems_tile0[i_kpr229 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr231 = 0U;
            for (; i_kpr231 < 1U; i_kpr231++) {
                uint32_t j_kpr232 = i_kpr231;
                vec_memcpy(elems_tile0 + (j_kpr232 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr232 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr233 = 0U;
            for (; i_kpr233 < 1U; i_kpr233++) {
                uint32_t j_kpr234 = i_kpr233;
                vec_memcpy(col_ind_tile0 + (j_kpr234 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr234 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr236 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr235 = 0U;
    for (; i_kpr235 < hi_kpr236; i_kpr235++) {
        uint32_t j_kpr237 = i_kpr235;
        elems_tile0[j_kpr237 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr237 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr237 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr237 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr238 = 0U;
    for (; i_kpr238 < 2U; i_kpr238++) {
        uint32_t j_kpr239 = i_kpr238;
        if (n_idx + j_kpr239 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr239 * 16U * 4U),
                out + j_kpr239 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x256x16
*/
static void
__hoisted_g_spmm_f32_64x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr240 = 0U;
        for (; i_kpr240 < 1U; i_kpr240++) {
            uint32_t j_kpr241 = i_kpr240;
            vec_memcpy(elems_tile0 + (j_kpr241 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr241 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr242 = 0U;
        for (; i_kpr242 < 1U; i_kpr242++) {
            uint32_t j_kpr243 = i_kpr242;
            vec_memcpy(col_ind_tile0 + (j_kpr243 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr243 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr244 = 0U;
        for (; i_kpr244 < to_; i_kpr244++)
            elems_tile0[i_kpr244 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr246 = 0U;
            for (; i_kpr246 < 1U; i_kpr246++) {
                uint32_t j_kpr247 = i_kpr246;
                vec_memcpy(elems_tile0 + (j_kpr247 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr247 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr248 = 0U;
            for (; i_kpr248 < 1U; i_kpr248++) {
                uint32_t j_kpr249 = i_kpr248;
                vec_memcpy(col_ind_tile0 + (j_kpr249 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr249 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr251 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr250 = 0U;
    for (; i_kpr250 < hi_kpr251; i_kpr250++) {
        uint32_t j_kpr252 = i_kpr250;
        elems_tile0[j_kpr252 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr252 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr252 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr252 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr253 = 0U;
    for (; i_kpr253 < 4U; i_kpr253++) {
        uint32_t j_kpr254 = i_kpr253;
        if (n_idx + j_kpr254 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr254 * 16U * 4U),
                out + j_kpr254 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x256x16_on
*/
static void
__hoisted_g_spmm_f32_64x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr255 = 0U;
        for (; i_kpr255 < 1U; i_kpr255++) {
            uint32_t j_kpr256 = i_kpr255;
            vec_memcpy(elems_tile0 + (j_kpr256 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr256 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr257 = 0U;
        for (; i_kpr257 < 1U; i_kpr257++) {
            uint32_t j_kpr258 = i_kpr257;
            vec_memcpy(col_ind_tile0 + (j_kpr258 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr258 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr259 = 0U;
        for (; i_kpr259 < to_; i_kpr259++)
            elems_tile0[i_kpr259 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr261 = 0U;
            for (; i_kpr261 < 1U; i_kpr261++) {
                uint32_t j_kpr262 = i_kpr261;
                vec_memcpy(elems_tile0 + (j_kpr262 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr262 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr263 = 0U;
            for (; i_kpr263 < 1U; i_kpr263++) {
                uint32_t j_kpr264 = i_kpr263;
                vec_memcpy(col_ind_tile0 + (j_kpr264 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr264 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr266 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr265 = 0U;
    for (; i_kpr265 < hi_kpr266; i_kpr265++) {
        uint32_t j_kpr267 = i_kpr265;
        elems_tile0[j_kpr267 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr267 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr267 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr267 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr268 = 0U;
    for (; i_kpr268 < 4U; i_kpr268++) {
        uint32_t j_kpr269 = i_kpr268;
        if (n_idx + j_kpr269 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr269 * 16U * 4U),
                out + j_kpr269 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x512x16
*/
static void
__hoisted_g_spmm_f32_64x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr270 = 0U;
        for (; i_kpr270 < 1U; i_kpr270++) {
            uint32_t j_kpr271 = i_kpr270;
            vec_memcpy(elems_tile0 + (j_kpr271 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr271 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr272 = 0U;
        for (; i_kpr272 < 1U; i_kpr272++) {
            uint32_t j_kpr273 = i_kpr272;
            vec_memcpy(col_ind_tile0 + (j_kpr273 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr273 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr274 = 0U;
        for (; i_kpr274 < to_; i_kpr274++)
            elems_tile0[i_kpr274 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr276 = 0U;
            for (; i_kpr276 < 1U; i_kpr276++) {
                uint32_t j_kpr277 = i_kpr276;
                vec_memcpy(elems_tile0 + (j_kpr277 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr277 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr278 = 0U;
            for (; i_kpr278 < 1U; i_kpr278++) {
                uint32_t j_kpr279 = i_kpr278;
                vec_memcpy(col_ind_tile0 + (j_kpr279 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr279 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr281 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr280 = 0U;
    for (; i_kpr280 < hi_kpr281; i_kpr280++) {
        uint32_t j_kpr282 = i_kpr280;
        elems_tile0[j_kpr282 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr282 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr282 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr282 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr283 = 0U;
    for (; i_kpr283 < 8U; i_kpr283++) {
        uint32_t j_kpr284 = i_kpr283;
        if (n_idx + j_kpr284 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr284 * 16U * 4U),
                out + j_kpr284 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x512x16_on
*/
static void
__hoisted_g_spmm_f32_64x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 64U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 64U) {
        uint32_t i_kpr285 = 0U;
        for (; i_kpr285 < 1U; i_kpr285++) {
            uint32_t j_kpr286 = i_kpr285;
            vec_memcpy(elems_tile0 + (j_kpr286 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr286 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr287 = 0U;
        for (; i_kpr287 < 1U; i_kpr287++) {
            uint32_t j_kpr288 = i_kpr287;
            vec_memcpy(col_ind_tile0 + (j_kpr288 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr288 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr289 = 0U;
        for (; i_kpr289 < to_; i_kpr289++)
            elems_tile0[i_kpr289 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 64U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr291 = 0U;
            for (; i_kpr291 < 1U; i_kpr291++) {
                uint32_t j_kpr292 = i_kpr291;
                vec_memcpy(elems_tile0 + (j_kpr292 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr292 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr293 = 0U;
            for (; i_kpr293 < 1U; i_kpr293++) {
                uint32_t j_kpr294 = i_kpr293;
                vec_memcpy(col_ind_tile0 + (j_kpr294 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr294 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 64U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr296 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr295 = 0U;
    for (; i_kpr295 < hi_kpr296; i_kpr295++) {
        uint32_t j_kpr297 = i_kpr295;
        elems_tile0[j_kpr297 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr297 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr297 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr297 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr298 = 0U;
    for (; i_kpr298 < 8U; i_kpr298++) {
        uint32_t j_kpr299 = i_kpr298;
        if (n_idx + j_kpr299 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr299 * 16U * 4U),
                out + j_kpr299 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x64x16
*/
static void
__hoisted_g_spmm_f32_128x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr300 = 0U;
        for (; i_kpr300 < 2U; i_kpr300++) {
            uint32_t j_kpr301 = i_kpr300;
            vec_memcpy(elems_tile0 + (j_kpr301 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr301 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr302 = 0U;
        for (; i_kpr302 < 2U; i_kpr302++) {
            uint32_t j_kpr303 = i_kpr302;
            vec_memcpy(col_ind_tile0 + (j_kpr303 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr303 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr304 = 0U;
        for (; i_kpr304 < to_; i_kpr304++)
            elems_tile0[i_kpr304 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr306 = 0U;
            for (; i_kpr306 < 2U; i_kpr306++) {
                uint32_t j_kpr307 = i_kpr306;
                vec_memcpy(elems_tile0 + (j_kpr307 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr307 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr308 = 0U;
            for (; i_kpr308 < 2U; i_kpr308++) {
                uint32_t j_kpr309 = i_kpr308;
                vec_memcpy(col_ind_tile0 + (j_kpr309 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr309 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr311 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr310 = 0U;
    for (; i_kpr310 < hi_kpr311; i_kpr310++) {
        uint32_t j_kpr312 = i_kpr310;
        elems_tile0[j_kpr312 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr312 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr312 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr312 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr313 = 0U;
    for (; i_kpr313 < 1U; i_kpr313++) {
        uint32_t j_kpr314 = i_kpr313;
        if (n_idx + j_kpr314 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr314 * 16U * 4U),
                out + j_kpr314 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x64x16_on
*/
static void
__hoisted_g_spmm_f32_128x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr315 = 0U;
        for (; i_kpr315 < 2U; i_kpr315++) {
            uint32_t j_kpr316 = i_kpr315;
            vec_memcpy(elems_tile0 + (j_kpr316 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr316 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr317 = 0U;
        for (; i_kpr317 < 2U; i_kpr317++) {
            uint32_t j_kpr318 = i_kpr317;
            vec_memcpy(col_ind_tile0 + (j_kpr318 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr318 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr319 = 0U;
        for (; i_kpr319 < to_; i_kpr319++)
            elems_tile0[i_kpr319 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr321 = 0U;
            for (; i_kpr321 < 2U; i_kpr321++) {
                uint32_t j_kpr322 = i_kpr321;
                vec_memcpy(elems_tile0 + (j_kpr322 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr322 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr323 = 0U;
            for (; i_kpr323 < 2U; i_kpr323++) {
                uint32_t j_kpr324 = i_kpr323;
                vec_memcpy(col_ind_tile0 + (j_kpr324 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr324 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr326 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr325 = 0U;
    for (; i_kpr325 < hi_kpr326; i_kpr325++) {
        uint32_t j_kpr327 = i_kpr325;
        elems_tile0[j_kpr327 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr327 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr327 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr327 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr328 = 0U;
    for (; i_kpr328 < 1U; i_kpr328++) {
        uint32_t j_kpr329 = i_kpr328;
        if (n_idx + j_kpr329 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr329 * 16U * 4U),
                out + j_kpr329 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x128x16
*/
static void
__hoisted_g_spmm_f32_128x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr330 = 0U;
        for (; i_kpr330 < 2U; i_kpr330++) {
            uint32_t j_kpr331 = i_kpr330;
            vec_memcpy(elems_tile0 + (j_kpr331 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr331 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr332 = 0U;
        for (; i_kpr332 < 2U; i_kpr332++) {
            uint32_t j_kpr333 = i_kpr332;
            vec_memcpy(col_ind_tile0 + (j_kpr333 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr333 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr334 = 0U;
        for (; i_kpr334 < to_; i_kpr334++)
            elems_tile0[i_kpr334 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr336 = 0U;
            for (; i_kpr336 < 2U; i_kpr336++) {
                uint32_t j_kpr337 = i_kpr336;
                vec_memcpy(elems_tile0 + (j_kpr337 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr337 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr338 = 0U;
            for (; i_kpr338 < 2U; i_kpr338++) {
                uint32_t j_kpr339 = i_kpr338;
                vec_memcpy(col_ind_tile0 + (j_kpr339 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr339 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr341 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr340 = 0U;
    for (; i_kpr340 < hi_kpr341; i_kpr340++) {
        uint32_t j_kpr342 = i_kpr340;
        elems_tile0[j_kpr342 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr342 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr342 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr342 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr343 = 0U;
    for (; i_kpr343 < 2U; i_kpr343++) {
        uint32_t j_kpr344 = i_kpr343;
        if (n_idx + j_kpr344 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr344 * 16U * 4U),
                out + j_kpr344 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x128x16_on
*/
static void
__hoisted_g_spmm_f32_128x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr345 = 0U;
        for (; i_kpr345 < 2U; i_kpr345++) {
            uint32_t j_kpr346 = i_kpr345;
            vec_memcpy(elems_tile0 + (j_kpr346 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr346 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr347 = 0U;
        for (; i_kpr347 < 2U; i_kpr347++) {
            uint32_t j_kpr348 = i_kpr347;
            vec_memcpy(col_ind_tile0 + (j_kpr348 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr348 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr349 = 0U;
        for (; i_kpr349 < to_; i_kpr349++)
            elems_tile0[i_kpr349 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr351 = 0U;
            for (; i_kpr351 < 2U; i_kpr351++) {
                uint32_t j_kpr352 = i_kpr351;
                vec_memcpy(elems_tile0 + (j_kpr352 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr352 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr353 = 0U;
            for (; i_kpr353 < 2U; i_kpr353++) {
                uint32_t j_kpr354 = i_kpr353;
                vec_memcpy(col_ind_tile0 + (j_kpr354 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr354 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr356 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr355 = 0U;
    for (; i_kpr355 < hi_kpr356; i_kpr355++) {
        uint32_t j_kpr357 = i_kpr355;
        elems_tile0[j_kpr357 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr357 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr357 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr357 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr358 = 0U;
    for (; i_kpr358 < 2U; i_kpr358++) {
        uint32_t j_kpr359 = i_kpr358;
        if (n_idx + j_kpr359 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr359 * 16U * 4U),
                out + j_kpr359 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x128x32
*/
static void
__hoisted_g_spmm_f32_128x128x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr360 = 0U;
        for (; i_kpr360 < 1U; i_kpr360++) {
            uint32_t j_kpr361 = i_kpr360;
            vec_memcpy(elems_tile0 + (j_kpr361 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr361 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr362 = 0U;
        for (; i_kpr362 < 1U; i_kpr362++) {
            uint32_t j_kpr363 = i_kpr362;
            vec_memcpy(col_ind_tile0 + (j_kpr363 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr363 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr364 = 0U;
        for (; i_kpr364 < to_; i_kpr364++)
            elems_tile0[i_kpr364 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr366 = 0U;
            for (; i_kpr366 < 1U; i_kpr366++) {
                uint32_t j_kpr367 = i_kpr366;
                vec_memcpy(elems_tile0 + (j_kpr367 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr367 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr368 = 0U;
            for (; i_kpr368 < 1U; i_kpr368++) {
                uint32_t j_kpr369 = i_kpr368;
                vec_memcpy(col_ind_tile0 + (j_kpr369 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr369 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr371 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr370 = 0U;
    for (; i_kpr370 < hi_kpr371; i_kpr370++) {
        uint32_t j_kpr372 = i_kpr370;
        elems_tile0[j_kpr372 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr372 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr372 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr372 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr373 = 0U;
    for (; i_kpr373 < 1U; i_kpr373++) {
        uint32_t j_kpr374 = i_kpr373;
        if (n_idx + j_kpr374 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr374 * 32U * 4U),
                out + j_kpr374 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x128x32_on
*/
static void
__hoisted_g_spmm_f32_128x128x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr375 = 0U;
        for (; i_kpr375 < 1U; i_kpr375++) {
            uint32_t j_kpr376 = i_kpr375;
            vec_memcpy(elems_tile0 + (j_kpr376 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr376 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr377 = 0U;
        for (; i_kpr377 < 1U; i_kpr377++) {
            uint32_t j_kpr378 = i_kpr377;
            vec_memcpy(col_ind_tile0 + (j_kpr378 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr378 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr379 = 0U;
        for (; i_kpr379 < to_; i_kpr379++)
            elems_tile0[i_kpr379 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr381 = 0U;
            for (; i_kpr381 < 1U; i_kpr381++) {
                uint32_t j_kpr382 = i_kpr381;
                vec_memcpy(elems_tile0 + (j_kpr382 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr382 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr383 = 0U;
            for (; i_kpr383 < 1U; i_kpr383++) {
                uint32_t j_kpr384 = i_kpr383;
                vec_memcpy(col_ind_tile0 + (j_kpr384 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr384 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr386 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr385 = 0U;
    for (; i_kpr385 < hi_kpr386; i_kpr385++) {
        uint32_t j_kpr387 = i_kpr385;
        elems_tile0[j_kpr387 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr387 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr387 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr387 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr388 = 0U;
    for (; i_kpr388 < 1U; i_kpr388++) {
        uint32_t j_kpr389 = i_kpr388;
        if (n_idx + j_kpr389 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr389 * 32U * 4U),
                out + j_kpr389 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x256x16
*/
static void
__hoisted_g_spmm_f32_128x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr390 = 0U;
        for (; i_kpr390 < 2U; i_kpr390++) {
            uint32_t j_kpr391 = i_kpr390;
            vec_memcpy(elems_tile0 + (j_kpr391 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr391 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr392 = 0U;
        for (; i_kpr392 < 2U; i_kpr392++) {
            uint32_t j_kpr393 = i_kpr392;
            vec_memcpy(col_ind_tile0 + (j_kpr393 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr393 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr394 = 0U;
        for (; i_kpr394 < to_; i_kpr394++)
            elems_tile0[i_kpr394 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr396 = 0U;
            for (; i_kpr396 < 2U; i_kpr396++) {
                uint32_t j_kpr397 = i_kpr396;
                vec_memcpy(elems_tile0 + (j_kpr397 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr397 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr398 = 0U;
            for (; i_kpr398 < 2U; i_kpr398++) {
                uint32_t j_kpr399 = i_kpr398;
                vec_memcpy(col_ind_tile0 + (j_kpr399 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr399 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr401 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr400 = 0U;
    for (; i_kpr400 < hi_kpr401; i_kpr400++) {
        uint32_t j_kpr402 = i_kpr400;
        elems_tile0[j_kpr402 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr402 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr402 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr402 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr403 = 0U;
    for (; i_kpr403 < 4U; i_kpr403++) {
        uint32_t j_kpr404 = i_kpr403;
        if (n_idx + j_kpr404 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr404 * 16U * 4U),
                out + j_kpr404 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x256x16_on
*/
static void
__hoisted_g_spmm_f32_128x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr405 = 0U;
        for (; i_kpr405 < 2U; i_kpr405++) {
            uint32_t j_kpr406 = i_kpr405;
            vec_memcpy(elems_tile0 + (j_kpr406 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr406 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr407 = 0U;
        for (; i_kpr407 < 2U; i_kpr407++) {
            uint32_t j_kpr408 = i_kpr407;
            vec_memcpy(col_ind_tile0 + (j_kpr408 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr408 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr409 = 0U;
        for (; i_kpr409 < to_; i_kpr409++)
            elems_tile0[i_kpr409 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr411 = 0U;
            for (; i_kpr411 < 2U; i_kpr411++) {
                uint32_t j_kpr412 = i_kpr411;
                vec_memcpy(elems_tile0 + (j_kpr412 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr412 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr413 = 0U;
            for (; i_kpr413 < 2U; i_kpr413++) {
                uint32_t j_kpr414 = i_kpr413;
                vec_memcpy(col_ind_tile0 + (j_kpr414 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr414 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr416 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr415 = 0U;
    for (; i_kpr415 < hi_kpr416; i_kpr415++) {
        uint32_t j_kpr417 = i_kpr415;
        elems_tile0[j_kpr417 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr417 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr417 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr417 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr418 = 0U;
    for (; i_kpr418 < 4U; i_kpr418++) {
        uint32_t j_kpr419 = i_kpr418;
        if (n_idx + j_kpr419 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr419 * 16U * 4U),
                out + j_kpr419 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x256x32
*/
static void
__hoisted_g_spmm_f32_128x256x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr420 = 0U;
        for (; i_kpr420 < 1U; i_kpr420++) {
            uint32_t j_kpr421 = i_kpr420;
            vec_memcpy(elems_tile0 + (j_kpr421 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr421 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr422 = 0U;
        for (; i_kpr422 < 1U; i_kpr422++) {
            uint32_t j_kpr423 = i_kpr422;
            vec_memcpy(col_ind_tile0 + (j_kpr423 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr423 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr424 = 0U;
        for (; i_kpr424 < to_; i_kpr424++)
            elems_tile0[i_kpr424 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr426 = 0U;
            for (; i_kpr426 < 1U; i_kpr426++) {
                uint32_t j_kpr427 = i_kpr426;
                vec_memcpy(elems_tile0 + (j_kpr427 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr427 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr428 = 0U;
            for (; i_kpr428 < 1U; i_kpr428++) {
                uint32_t j_kpr429 = i_kpr428;
                vec_memcpy(col_ind_tile0 + (j_kpr429 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr429 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr431 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr430 = 0U;
    for (; i_kpr430 < hi_kpr431; i_kpr430++) {
        uint32_t j_kpr432 = i_kpr430;
        elems_tile0[j_kpr432 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr432 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr432 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr432 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr433 = 0U;
    for (; i_kpr433 < 2U; i_kpr433++) {
        uint32_t j_kpr434 = i_kpr433;
        if (n_idx + j_kpr434 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr434 * 32U * 4U),
                out + j_kpr434 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x256x32_on
*/
static void
__hoisted_g_spmm_f32_128x256x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr435 = 0U;
        for (; i_kpr435 < 1U; i_kpr435++) {
            uint32_t j_kpr436 = i_kpr435;
            vec_memcpy(elems_tile0 + (j_kpr436 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr436 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr437 = 0U;
        for (; i_kpr437 < 1U; i_kpr437++) {
            uint32_t j_kpr438 = i_kpr437;
            vec_memcpy(col_ind_tile0 + (j_kpr438 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr438 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr439 = 0U;
        for (; i_kpr439 < to_; i_kpr439++)
            elems_tile0[i_kpr439 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr441 = 0U;
            for (; i_kpr441 < 1U; i_kpr441++) {
                uint32_t j_kpr442 = i_kpr441;
                vec_memcpy(elems_tile0 + (j_kpr442 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr442 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr443 = 0U;
            for (; i_kpr443 < 1U; i_kpr443++) {
                uint32_t j_kpr444 = i_kpr443;
                vec_memcpy(col_ind_tile0 + (j_kpr444 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr444 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr446 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr445 = 0U;
    for (; i_kpr445 < hi_kpr446; i_kpr445++) {
        uint32_t j_kpr447 = i_kpr445;
        elems_tile0[j_kpr447 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr447 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr447 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr447 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr448 = 0U;
    for (; i_kpr448 < 2U; i_kpr448++) {
        uint32_t j_kpr449 = i_kpr448;
        if (n_idx + j_kpr449 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr449 * 32U * 4U),
                out + j_kpr449 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x512x16
*/
static void
__hoisted_g_spmm_f32_128x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr450 = 0U;
        for (; i_kpr450 < 2U; i_kpr450++) {
            uint32_t j_kpr451 = i_kpr450;
            vec_memcpy(elems_tile0 + (j_kpr451 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr451 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr452 = 0U;
        for (; i_kpr452 < 2U; i_kpr452++) {
            uint32_t j_kpr453 = i_kpr452;
            vec_memcpy(col_ind_tile0 + (j_kpr453 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr453 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr454 = 0U;
        for (; i_kpr454 < to_; i_kpr454++)
            elems_tile0[i_kpr454 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr456 = 0U;
            for (; i_kpr456 < 2U; i_kpr456++) {
                uint32_t j_kpr457 = i_kpr456;
                vec_memcpy(elems_tile0 + (j_kpr457 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr457 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr458 = 0U;
            for (; i_kpr458 < 2U; i_kpr458++) {
                uint32_t j_kpr459 = i_kpr458;
                vec_memcpy(col_ind_tile0 + (j_kpr459 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr459 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr461 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr460 = 0U;
    for (; i_kpr460 < hi_kpr461; i_kpr460++) {
        uint32_t j_kpr462 = i_kpr460;
        elems_tile0[j_kpr462 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr462 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr462 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr462 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr463 = 0U;
    for (; i_kpr463 < 8U; i_kpr463++) {
        uint32_t j_kpr464 = i_kpr463;
        if (n_idx + j_kpr464 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr464 * 16U * 4U),
                out + j_kpr464 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x512x16_on
*/
static void
__hoisted_g_spmm_f32_128x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr465 = 0U;
        for (; i_kpr465 < 2U; i_kpr465++) {
            uint32_t j_kpr466 = i_kpr465;
            vec_memcpy(elems_tile0 + (j_kpr466 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr466 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr467 = 0U;
        for (; i_kpr467 < 2U; i_kpr467++) {
            uint32_t j_kpr468 = i_kpr467;
            vec_memcpy(col_ind_tile0 + (j_kpr468 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr468 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr469 = 0U;
        for (; i_kpr469 < to_; i_kpr469++)
            elems_tile0[i_kpr469 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr471 = 0U;
            for (; i_kpr471 < 2U; i_kpr471++) {
                uint32_t j_kpr472 = i_kpr471;
                vec_memcpy(elems_tile0 + (j_kpr472 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr472 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr473 = 0U;
            for (; i_kpr473 < 2U; i_kpr473++) {
                uint32_t j_kpr474 = i_kpr473;
                vec_memcpy(col_ind_tile0 + (j_kpr474 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr474 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr476 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr475 = 0U;
    for (; i_kpr475 < hi_kpr476; i_kpr475++) {
        uint32_t j_kpr477 = i_kpr475;
        elems_tile0[j_kpr477 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr477 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr477 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr477 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr478 = 0U;
    for (; i_kpr478 < 8U; i_kpr478++) {
        uint32_t j_kpr479 = i_kpr478;
        if (n_idx + j_kpr479 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr479 * 16U * 4U),
                out + j_kpr479 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x512x32
*/
static void
__hoisted_g_spmm_f32_128x512x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr480 = 0U;
        for (; i_kpr480 < 1U; i_kpr480++) {
            uint32_t j_kpr481 = i_kpr480;
            vec_memcpy(elems_tile0 + (j_kpr481 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr481 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr482 = 0U;
        for (; i_kpr482 < 1U; i_kpr482++) {
            uint32_t j_kpr483 = i_kpr482;
            vec_memcpy(col_ind_tile0 + (j_kpr483 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr483 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr484 = 0U;
        for (; i_kpr484 < to_; i_kpr484++)
            elems_tile0[i_kpr484 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr486 = 0U;
            for (; i_kpr486 < 1U; i_kpr486++) {
                uint32_t j_kpr487 = i_kpr486;
                vec_memcpy(elems_tile0 + (j_kpr487 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr487 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr488 = 0U;
            for (; i_kpr488 < 1U; i_kpr488++) {
                uint32_t j_kpr489 = i_kpr488;
                vec_memcpy(col_ind_tile0 + (j_kpr489 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr489 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr491 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr490 = 0U;
    for (; i_kpr490 < hi_kpr491; i_kpr490++) {
        uint32_t j_kpr492 = i_kpr490;
        elems_tile0[j_kpr492 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr492 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr492 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr492 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr493 = 0U;
    for (; i_kpr493 < 4U; i_kpr493++) {
        uint32_t j_kpr494 = i_kpr493;
        if (n_idx + j_kpr494 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr494 * 32U * 4U),
                out + j_kpr494 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x512x32_on
*/
static void
__hoisted_g_spmm_f32_128x512x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 128U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 128U) {
        uint32_t i_kpr495 = 0U;
        for (; i_kpr495 < 1U; i_kpr495++) {
            uint32_t j_kpr496 = i_kpr495;
            vec_memcpy(elems_tile0 + (j_kpr496 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr496 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr497 = 0U;
        for (; i_kpr497 < 1U; i_kpr497++) {
            uint32_t j_kpr498 = i_kpr497;
            vec_memcpy(col_ind_tile0 + (j_kpr498 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr498 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr499 = 0U;
        for (; i_kpr499 < to_; i_kpr499++)
            elems_tile0[i_kpr499 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 128U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr501 = 0U;
            for (; i_kpr501 < 1U; i_kpr501++) {
                uint32_t j_kpr502 = i_kpr501;
                vec_memcpy(elems_tile0 + (j_kpr502 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr502 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr503 = 0U;
            for (; i_kpr503 < 1U; i_kpr503++) {
                uint32_t j_kpr504 = i_kpr503;
                vec_memcpy(col_ind_tile0 + (j_kpr504 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr504 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 128U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr506 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr505 = 0U;
    for (; i_kpr505 < hi_kpr506; i_kpr505++) {
        uint32_t j_kpr507 = i_kpr505;
        elems_tile0[j_kpr507 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr507 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr507 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr507 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr508 = 0U;
    for (; i_kpr508 < 4U; i_kpr508++) {
        uint32_t j_kpr509 = i_kpr508;
        if (n_idx + j_kpr509 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr509 * 32U * 4U),
                out + j_kpr509 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x64x16
*/
static void
__hoisted_g_spmm_f32_256x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr510 = 0U;
        for (; i_kpr510 < 4U; i_kpr510++) {
            uint32_t j_kpr511 = i_kpr510;
            vec_memcpy(elems_tile0 + (j_kpr511 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr511 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr512 = 0U;
        for (; i_kpr512 < 4U; i_kpr512++) {
            uint32_t j_kpr513 = i_kpr512;
            vec_memcpy(col_ind_tile0 + (j_kpr513 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr513 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr514 = 0U;
        for (; i_kpr514 < to_; i_kpr514++)
            elems_tile0[i_kpr514 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr516 = 0U;
            for (; i_kpr516 < 4U; i_kpr516++) {
                uint32_t j_kpr517 = i_kpr516;
                vec_memcpy(elems_tile0 + (j_kpr517 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr517 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr518 = 0U;
            for (; i_kpr518 < 4U; i_kpr518++) {
                uint32_t j_kpr519 = i_kpr518;
                vec_memcpy(col_ind_tile0 + (j_kpr519 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr519 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr521 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr520 = 0U;
    for (; i_kpr520 < hi_kpr521; i_kpr520++) {
        uint32_t j_kpr522 = i_kpr520;
        elems_tile0[j_kpr522 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr522 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr522 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr522 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr523 = 0U;
    for (; i_kpr523 < 1U; i_kpr523++) {
        uint32_t j_kpr524 = i_kpr523;
        if (n_idx + j_kpr524 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr524 * 16U * 4U),
                out + j_kpr524 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x64x16_on
*/
static void
__hoisted_g_spmm_f32_256x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr525 = 0U;
        for (; i_kpr525 < 4U; i_kpr525++) {
            uint32_t j_kpr526 = i_kpr525;
            vec_memcpy(elems_tile0 + (j_kpr526 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr526 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr527 = 0U;
        for (; i_kpr527 < 4U; i_kpr527++) {
            uint32_t j_kpr528 = i_kpr527;
            vec_memcpy(col_ind_tile0 + (j_kpr528 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr528 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr529 = 0U;
        for (; i_kpr529 < to_; i_kpr529++)
            elems_tile0[i_kpr529 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr531 = 0U;
            for (; i_kpr531 < 4U; i_kpr531++) {
                uint32_t j_kpr532 = i_kpr531;
                vec_memcpy(elems_tile0 + (j_kpr532 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr532 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr533 = 0U;
            for (; i_kpr533 < 4U; i_kpr533++) {
                uint32_t j_kpr534 = i_kpr533;
                vec_memcpy(col_ind_tile0 + (j_kpr534 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr534 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr536 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr535 = 0U;
    for (; i_kpr535 < hi_kpr536; i_kpr535++) {
        uint32_t j_kpr537 = i_kpr535;
        elems_tile0[j_kpr537 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr537 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr537 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr537 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr538 = 0U;
    for (; i_kpr538 < 1U; i_kpr538++) {
        uint32_t j_kpr539 = i_kpr538;
        if (n_idx + j_kpr539 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr539 * 16U * 4U),
                out + j_kpr539 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x128x16
*/
static void
__hoisted_g_spmm_f32_256x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr540 = 0U;
        for (; i_kpr540 < 4U; i_kpr540++) {
            uint32_t j_kpr541 = i_kpr540;
            vec_memcpy(elems_tile0 + (j_kpr541 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr541 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr542 = 0U;
        for (; i_kpr542 < 4U; i_kpr542++) {
            uint32_t j_kpr543 = i_kpr542;
            vec_memcpy(col_ind_tile0 + (j_kpr543 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr543 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr544 = 0U;
        for (; i_kpr544 < to_; i_kpr544++)
            elems_tile0[i_kpr544 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr546 = 0U;
            for (; i_kpr546 < 4U; i_kpr546++) {
                uint32_t j_kpr547 = i_kpr546;
                vec_memcpy(elems_tile0 + (j_kpr547 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr547 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr548 = 0U;
            for (; i_kpr548 < 4U; i_kpr548++) {
                uint32_t j_kpr549 = i_kpr548;
                vec_memcpy(col_ind_tile0 + (j_kpr549 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr549 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr551 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr550 = 0U;
    for (; i_kpr550 < hi_kpr551; i_kpr550++) {
        uint32_t j_kpr552 = i_kpr550;
        elems_tile0[j_kpr552 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr552 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr552 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr552 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr553 = 0U;
    for (; i_kpr553 < 2U; i_kpr553++) {
        uint32_t j_kpr554 = i_kpr553;
        if (n_idx + j_kpr554 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr554 * 16U * 4U),
                out + j_kpr554 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x128x16_on
*/
static void
__hoisted_g_spmm_f32_256x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr555 = 0U;
        for (; i_kpr555 < 4U; i_kpr555++) {
            uint32_t j_kpr556 = i_kpr555;
            vec_memcpy(elems_tile0 + (j_kpr556 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr556 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr557 = 0U;
        for (; i_kpr557 < 4U; i_kpr557++) {
            uint32_t j_kpr558 = i_kpr557;
            vec_memcpy(col_ind_tile0 + (j_kpr558 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr558 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr559 = 0U;
        for (; i_kpr559 < to_; i_kpr559++)
            elems_tile0[i_kpr559 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr561 = 0U;
            for (; i_kpr561 < 4U; i_kpr561++) {
                uint32_t j_kpr562 = i_kpr561;
                vec_memcpy(elems_tile0 + (j_kpr562 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr562 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr563 = 0U;
            for (; i_kpr563 < 4U; i_kpr563++) {
                uint32_t j_kpr564 = i_kpr563;
                vec_memcpy(col_ind_tile0 + (j_kpr564 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr564 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr566 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr565 = 0U;
    for (; i_kpr565 < hi_kpr566; i_kpr565++) {
        uint32_t j_kpr567 = i_kpr565;
        elems_tile0[j_kpr567 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr567 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr567 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr567 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr568 = 0U;
    for (; i_kpr568 < 2U; i_kpr568++) {
        uint32_t j_kpr569 = i_kpr568;
        if (n_idx + j_kpr569 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr569 * 16U * 4U),
                out + j_kpr569 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x128x32
*/
static void
__hoisted_g_spmm_f32_256x128x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr570 = 0U;
        for (; i_kpr570 < 2U; i_kpr570++) {
            uint32_t j_kpr571 = i_kpr570;
            vec_memcpy(elems_tile0 + (j_kpr571 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr571 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr572 = 0U;
        for (; i_kpr572 < 2U; i_kpr572++) {
            uint32_t j_kpr573 = i_kpr572;
            vec_memcpy(col_ind_tile0 + (j_kpr573 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr573 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr574 = 0U;
        for (; i_kpr574 < to_; i_kpr574++)
            elems_tile0[i_kpr574 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr576 = 0U;
            for (; i_kpr576 < 2U; i_kpr576++) {
                uint32_t j_kpr577 = i_kpr576;
                vec_memcpy(elems_tile0 + (j_kpr577 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr577 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr578 = 0U;
            for (; i_kpr578 < 2U; i_kpr578++) {
                uint32_t j_kpr579 = i_kpr578;
                vec_memcpy(col_ind_tile0 + (j_kpr579 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr579 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr581 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr580 = 0U;
    for (; i_kpr580 < hi_kpr581; i_kpr580++) {
        uint32_t j_kpr582 = i_kpr580;
        elems_tile0[j_kpr582 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr582 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr582 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr582 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr583 = 0U;
    for (; i_kpr583 < 1U; i_kpr583++) {
        uint32_t j_kpr584 = i_kpr583;
        if (n_idx + j_kpr584 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr584 * 32U * 4U),
                out + j_kpr584 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x128x32_on
*/
static void
__hoisted_g_spmm_f32_256x128x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr585 = 0U;
        for (; i_kpr585 < 2U; i_kpr585++) {
            uint32_t j_kpr586 = i_kpr585;
            vec_memcpy(elems_tile0 + (j_kpr586 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr586 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr587 = 0U;
        for (; i_kpr587 < 2U; i_kpr587++) {
            uint32_t j_kpr588 = i_kpr587;
            vec_memcpy(col_ind_tile0 + (j_kpr588 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr588 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr589 = 0U;
        for (; i_kpr589 < to_; i_kpr589++)
            elems_tile0[i_kpr589 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr591 = 0U;
            for (; i_kpr591 < 2U; i_kpr591++) {
                uint32_t j_kpr592 = i_kpr591;
                vec_memcpy(elems_tile0 + (j_kpr592 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr592 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr593 = 0U;
            for (; i_kpr593 < 2U; i_kpr593++) {
                uint32_t j_kpr594 = i_kpr593;
                vec_memcpy(col_ind_tile0 + (j_kpr594 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr594 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr596 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr595 = 0U;
    for (; i_kpr595 < hi_kpr596; i_kpr595++) {
        uint32_t j_kpr597 = i_kpr595;
        elems_tile0[j_kpr597 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr597 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr597 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr597 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr598 = 0U;
    for (; i_kpr598 < 1U; i_kpr598++) {
        uint32_t j_kpr599 = i_kpr598;
        if (n_idx + j_kpr599 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr599 * 32U * 4U),
                out + j_kpr599 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x256x16
*/
static void
__hoisted_g_spmm_f32_256x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr600 = 0U;
        for (; i_kpr600 < 4U; i_kpr600++) {
            uint32_t j_kpr601 = i_kpr600;
            vec_memcpy(elems_tile0 + (j_kpr601 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr601 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr602 = 0U;
        for (; i_kpr602 < 4U; i_kpr602++) {
            uint32_t j_kpr603 = i_kpr602;
            vec_memcpy(col_ind_tile0 + (j_kpr603 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr603 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr604 = 0U;
        for (; i_kpr604 < to_; i_kpr604++)
            elems_tile0[i_kpr604 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr606 = 0U;
            for (; i_kpr606 < 4U; i_kpr606++) {
                uint32_t j_kpr607 = i_kpr606;
                vec_memcpy(elems_tile0 + (j_kpr607 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr607 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr608 = 0U;
            for (; i_kpr608 < 4U; i_kpr608++) {
                uint32_t j_kpr609 = i_kpr608;
                vec_memcpy(col_ind_tile0 + (j_kpr609 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr609 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr611 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr610 = 0U;
    for (; i_kpr610 < hi_kpr611; i_kpr610++) {
        uint32_t j_kpr612 = i_kpr610;
        elems_tile0[j_kpr612 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr612 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr612 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr612 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr613 = 0U;
    for (; i_kpr613 < 4U; i_kpr613++) {
        uint32_t j_kpr614 = i_kpr613;
        if (n_idx + j_kpr614 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr614 * 16U * 4U),
                out + j_kpr614 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x256x16_on
*/
static void
__hoisted_g_spmm_f32_256x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr615 = 0U;
        for (; i_kpr615 < 4U; i_kpr615++) {
            uint32_t j_kpr616 = i_kpr615;
            vec_memcpy(elems_tile0 + (j_kpr616 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr616 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr617 = 0U;
        for (; i_kpr617 < 4U; i_kpr617++) {
            uint32_t j_kpr618 = i_kpr617;
            vec_memcpy(col_ind_tile0 + (j_kpr618 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr618 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr619 = 0U;
        for (; i_kpr619 < to_; i_kpr619++)
            elems_tile0[i_kpr619 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr621 = 0U;
            for (; i_kpr621 < 4U; i_kpr621++) {
                uint32_t j_kpr622 = i_kpr621;
                vec_memcpy(elems_tile0 + (j_kpr622 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr622 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr623 = 0U;
            for (; i_kpr623 < 4U; i_kpr623++) {
                uint32_t j_kpr624 = i_kpr623;
                vec_memcpy(col_ind_tile0 + (j_kpr624 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr624 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr626 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr625 = 0U;
    for (; i_kpr625 < hi_kpr626; i_kpr625++) {
        uint32_t j_kpr627 = i_kpr625;
        elems_tile0[j_kpr627 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr627 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr627 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr627 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr628 = 0U;
    for (; i_kpr628 < 4U; i_kpr628++) {
        uint32_t j_kpr629 = i_kpr628;
        if (n_idx + j_kpr629 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr629 * 16U * 4U),
                out + j_kpr629 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x256x32
*/
static void
__hoisted_g_spmm_f32_256x256x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr630 = 0U;
        for (; i_kpr630 < 2U; i_kpr630++) {
            uint32_t j_kpr631 = i_kpr630;
            vec_memcpy(elems_tile0 + (j_kpr631 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr631 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr632 = 0U;
        for (; i_kpr632 < 2U; i_kpr632++) {
            uint32_t j_kpr633 = i_kpr632;
            vec_memcpy(col_ind_tile0 + (j_kpr633 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr633 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr634 = 0U;
        for (; i_kpr634 < to_; i_kpr634++)
            elems_tile0[i_kpr634 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr636 = 0U;
            for (; i_kpr636 < 2U; i_kpr636++) {
                uint32_t j_kpr637 = i_kpr636;
                vec_memcpy(elems_tile0 + (j_kpr637 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr637 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr638 = 0U;
            for (; i_kpr638 < 2U; i_kpr638++) {
                uint32_t j_kpr639 = i_kpr638;
                vec_memcpy(col_ind_tile0 + (j_kpr639 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr639 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr641 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr640 = 0U;
    for (; i_kpr640 < hi_kpr641; i_kpr640++) {
        uint32_t j_kpr642 = i_kpr640;
        elems_tile0[j_kpr642 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr642 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr642 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr642 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr643 = 0U;
    for (; i_kpr643 < 2U; i_kpr643++) {
        uint32_t j_kpr644 = i_kpr643;
        if (n_idx + j_kpr644 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr644 * 32U * 4U),
                out + j_kpr644 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x256x32_on
*/
static void
__hoisted_g_spmm_f32_256x256x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr645 = 0U;
        for (; i_kpr645 < 2U; i_kpr645++) {
            uint32_t j_kpr646 = i_kpr645;
            vec_memcpy(elems_tile0 + (j_kpr646 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr646 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr647 = 0U;
        for (; i_kpr647 < 2U; i_kpr647++) {
            uint32_t j_kpr648 = i_kpr647;
            vec_memcpy(col_ind_tile0 + (j_kpr648 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr648 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr649 = 0U;
        for (; i_kpr649 < to_; i_kpr649++)
            elems_tile0[i_kpr649 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr651 = 0U;
            for (; i_kpr651 < 2U; i_kpr651++) {
                uint32_t j_kpr652 = i_kpr651;
                vec_memcpy(elems_tile0 + (j_kpr652 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr652 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr653 = 0U;
            for (; i_kpr653 < 2U; i_kpr653++) {
                uint32_t j_kpr654 = i_kpr653;
                vec_memcpy(col_ind_tile0 + (j_kpr654 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr654 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr656 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr655 = 0U;
    for (; i_kpr655 < hi_kpr656; i_kpr655++) {
        uint32_t j_kpr657 = i_kpr655;
        elems_tile0[j_kpr657 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr657 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr657 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr657 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr658 = 0U;
    for (; i_kpr658 < 2U; i_kpr658++) {
        uint32_t j_kpr659 = i_kpr658;
        if (n_idx + j_kpr659 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr659 * 32U * 4U),
                out + j_kpr659 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x256x64
*/
static void
__hoisted_g_spmm_f32_256x256x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr660 = 0U;
        for (; i_kpr660 < 1U; i_kpr660++) {
            uint32_t j_kpr661 = i_kpr660;
            vec_memcpy(elems_tile0 + (j_kpr661 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr661 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr662 = 0U;
        for (; i_kpr662 < 1U; i_kpr662++) {
            uint32_t j_kpr663 = i_kpr662;
            vec_memcpy(col_ind_tile0 + (j_kpr663 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr663 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr664 = 0U;
        for (; i_kpr664 < to_; i_kpr664++)
            elems_tile0[i_kpr664 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
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
            uint32_t i_kpr666 = 0U;
            for (; i_kpr666 < 1U; i_kpr666++) {
                uint32_t j_kpr667 = i_kpr666;
                vec_memcpy(elems_tile0 + (j_kpr667 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr667 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr668 = 0U;
            for (; i_kpr668 < 1U; i_kpr668++) {
                uint32_t j_kpr669 = i_kpr668;
                vec_memcpy(col_ind_tile0 + (j_kpr669 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr669 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr671 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr670 = 0U;
    for (; i_kpr670 < hi_kpr671; i_kpr670++) {
        uint32_t j_kpr672 = i_kpr670;
        elems_tile0[j_kpr672 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr672 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr672 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr672 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr673 = 0U;
    for (; i_kpr673 < 1U; i_kpr673++) {
        uint32_t j_kpr674 = i_kpr673;
        if (n_idx + j_kpr674 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr674 * 64U * 4U),
                out + j_kpr674 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x256x64_on
*/
static void
__hoisted_g_spmm_f32_256x256x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr675 = 0U;
        for (; i_kpr675 < 1U; i_kpr675++) {
            uint32_t j_kpr676 = i_kpr675;
            vec_memcpy(elems_tile0 + (j_kpr676 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr676 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr677 = 0U;
        for (; i_kpr677 < 1U; i_kpr677++) {
            uint32_t j_kpr678 = i_kpr677;
            vec_memcpy(col_ind_tile0 + (j_kpr678 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr678 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr679 = 0U;
        for (; i_kpr679 < to_; i_kpr679++)
            elems_tile0[i_kpr679 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
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
            uint32_t i_kpr681 = 0U;
            for (; i_kpr681 < 1U; i_kpr681++) {
                uint32_t j_kpr682 = i_kpr681;
                vec_memcpy(elems_tile0 + (j_kpr682 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr682 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr683 = 0U;
            for (; i_kpr683 < 1U; i_kpr683++) {
                uint32_t j_kpr684 = i_kpr683;
                vec_memcpy(col_ind_tile0 + (j_kpr684 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr684 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr686 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr685 = 0U;
    for (; i_kpr685 < hi_kpr686; i_kpr685++) {
        uint32_t j_kpr687 = i_kpr685;
        elems_tile0[j_kpr687 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr687 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr687 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr687 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr688 = 0U;
    for (; i_kpr688 < 1U; i_kpr688++) {
        uint32_t j_kpr689 = i_kpr688;
        if (n_idx + j_kpr689 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr689 * 64U * 4U),
                out + j_kpr689 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x512x16
*/
static void
__hoisted_g_spmm_f32_256x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr690 = 0U;
        for (; i_kpr690 < 4U; i_kpr690++) {
            uint32_t j_kpr691 = i_kpr690;
            vec_memcpy(elems_tile0 + (j_kpr691 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr691 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr692 = 0U;
        for (; i_kpr692 < 4U; i_kpr692++) {
            uint32_t j_kpr693 = i_kpr692;
            vec_memcpy(col_ind_tile0 + (j_kpr693 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr693 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr694 = 0U;
        for (; i_kpr694 < to_; i_kpr694++)
            elems_tile0[i_kpr694 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr696 = 0U;
            for (; i_kpr696 < 4U; i_kpr696++) {
                uint32_t j_kpr697 = i_kpr696;
                vec_memcpy(elems_tile0 + (j_kpr697 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr697 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr698 = 0U;
            for (; i_kpr698 < 4U; i_kpr698++) {
                uint32_t j_kpr699 = i_kpr698;
                vec_memcpy(col_ind_tile0 + (j_kpr699 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr699 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr701 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr700 = 0U;
    for (; i_kpr700 < hi_kpr701; i_kpr700++) {
        uint32_t j_kpr702 = i_kpr700;
        elems_tile0[j_kpr702 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr702 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr702 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr702 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr703 = 0U;
    for (; i_kpr703 < 8U; i_kpr703++) {
        uint32_t j_kpr704 = i_kpr703;
        if (n_idx + j_kpr704 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr704 * 16U * 4U),
                out + j_kpr704 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x512x16_on
*/
static void
__hoisted_g_spmm_f32_256x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr705 = 0U;
        for (; i_kpr705 < 4U; i_kpr705++) {
            uint32_t j_kpr706 = i_kpr705;
            vec_memcpy(elems_tile0 + (j_kpr706 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr706 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr707 = 0U;
        for (; i_kpr707 < 4U; i_kpr707++) {
            uint32_t j_kpr708 = i_kpr707;
            vec_memcpy(col_ind_tile0 + (j_kpr708 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr708 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr709 = 0U;
        for (; i_kpr709 < to_; i_kpr709++)
            elems_tile0[i_kpr709 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr711 = 0U;
            for (; i_kpr711 < 4U; i_kpr711++) {
                uint32_t j_kpr712 = i_kpr711;
                vec_memcpy(elems_tile0 + (j_kpr712 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr712 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr713 = 0U;
            for (; i_kpr713 < 4U; i_kpr713++) {
                uint32_t j_kpr714 = i_kpr713;
                vec_memcpy(col_ind_tile0 + (j_kpr714 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr714 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr716 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr715 = 0U;
    for (; i_kpr715 < hi_kpr716; i_kpr715++) {
        uint32_t j_kpr717 = i_kpr715;
        elems_tile0[j_kpr717 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr717 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr717 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr717 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr718 = 0U;
    for (; i_kpr718 < 8U; i_kpr718++) {
        uint32_t j_kpr719 = i_kpr718;
        if (n_idx + j_kpr719 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr719 * 16U * 4U),
                out + j_kpr719 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x512x32
*/
static void
__hoisted_g_spmm_f32_256x512x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr720 = 0U;
        for (; i_kpr720 < 2U; i_kpr720++) {
            uint32_t j_kpr721 = i_kpr720;
            vec_memcpy(elems_tile0 + (j_kpr721 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr721 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr722 = 0U;
        for (; i_kpr722 < 2U; i_kpr722++) {
            uint32_t j_kpr723 = i_kpr722;
            vec_memcpy(col_ind_tile0 + (j_kpr723 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr723 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr724 = 0U;
        for (; i_kpr724 < to_; i_kpr724++)
            elems_tile0[i_kpr724 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr726 = 0U;
            for (; i_kpr726 < 2U; i_kpr726++) {
                uint32_t j_kpr727 = i_kpr726;
                vec_memcpy(elems_tile0 + (j_kpr727 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr727 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr728 = 0U;
            for (; i_kpr728 < 2U; i_kpr728++) {
                uint32_t j_kpr729 = i_kpr728;
                vec_memcpy(col_ind_tile0 + (j_kpr729 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr729 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr731 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr730 = 0U;
    for (; i_kpr730 < hi_kpr731; i_kpr730++) {
        uint32_t j_kpr732 = i_kpr730;
        elems_tile0[j_kpr732 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr732 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr732 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr732 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr733 = 0U;
    for (; i_kpr733 < 4U; i_kpr733++) {
        uint32_t j_kpr734 = i_kpr733;
        if (n_idx + j_kpr734 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr734 * 32U * 4U),
                out + j_kpr734 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x512x32_on
*/
static void
__hoisted_g_spmm_f32_256x512x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr735 = 0U;
        for (; i_kpr735 < 2U; i_kpr735++) {
            uint32_t j_kpr736 = i_kpr735;
            vec_memcpy(elems_tile0 + (j_kpr736 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr736 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr737 = 0U;
        for (; i_kpr737 < 2U; i_kpr737++) {
            uint32_t j_kpr738 = i_kpr737;
            vec_memcpy(col_ind_tile0 + (j_kpr738 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr738 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr739 = 0U;
        for (; i_kpr739 < to_; i_kpr739++)
            elems_tile0[i_kpr739 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr741 = 0U;
            for (; i_kpr741 < 2U; i_kpr741++) {
                uint32_t j_kpr742 = i_kpr741;
                vec_memcpy(elems_tile0 + (j_kpr742 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr742 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr743 = 0U;
            for (; i_kpr743 < 2U; i_kpr743++) {
                uint32_t j_kpr744 = i_kpr743;
                vec_memcpy(col_ind_tile0 + (j_kpr744 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr744 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr746 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr745 = 0U;
    for (; i_kpr745 < hi_kpr746; i_kpr745++) {
        uint32_t j_kpr747 = i_kpr745;
        elems_tile0[j_kpr747 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr747 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr747 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr747 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr748 = 0U;
    for (; i_kpr748 < 4U; i_kpr748++) {
        uint32_t j_kpr749 = i_kpr748;
        if (n_idx + j_kpr749 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr749 * 32U * 4U),
                out + j_kpr749 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x512x64
*/
static void
__hoisted_g_spmm_f32_256x512x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr750 = 0U;
        for (; i_kpr750 < 1U; i_kpr750++) {
            uint32_t j_kpr751 = i_kpr750;
            vec_memcpy(elems_tile0 + (j_kpr751 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr751 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr752 = 0U;
        for (; i_kpr752 < 1U; i_kpr752++) {
            uint32_t j_kpr753 = i_kpr752;
            vec_memcpy(col_ind_tile0 + (j_kpr753 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr753 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr754 = 0U;
        for (; i_kpr754 < to_; i_kpr754++)
            elems_tile0[i_kpr754 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr756 = 0U;
            for (; i_kpr756 < 1U; i_kpr756++) {
                uint32_t j_kpr757 = i_kpr756;
                vec_memcpy(elems_tile0 + (j_kpr757 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr757 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr758 = 0U;
            for (; i_kpr758 < 1U; i_kpr758++) {
                uint32_t j_kpr759 = i_kpr758;
                vec_memcpy(col_ind_tile0 + (j_kpr759 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr759 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 64U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr761 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr760 = 0U;
    for (; i_kpr760 < hi_kpr761; i_kpr760++) {
        uint32_t j_kpr762 = i_kpr760;
        elems_tile0[j_kpr762 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr762 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr762 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr762 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr763 = 0U;
    for (; i_kpr763 < 2U; i_kpr763++) {
        uint32_t j_kpr764 = i_kpr763;
        if (n_idx + j_kpr764 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr764 * 64U * 4U),
                out + j_kpr764 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x512x64_on
*/
static void
__hoisted_g_spmm_f32_256x512x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 256U) {
        uint32_t i_kpr765 = 0U;
        for (; i_kpr765 < 1U; i_kpr765++) {
            uint32_t j_kpr766 = i_kpr765;
            vec_memcpy(elems_tile0 + (j_kpr766 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr766 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr767 = 0U;
        for (; i_kpr767 < 1U; i_kpr767++) {
            uint32_t j_kpr768 = i_kpr767;
            vec_memcpy(col_ind_tile0 + (j_kpr768 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr768 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr769 = 0U;
        for (; i_kpr769 < to_; i_kpr769++)
            elems_tile0[i_kpr769 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 256U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr771 = 0U;
            for (; i_kpr771 < 1U; i_kpr771++) {
                uint32_t j_kpr772 = i_kpr771;
                vec_memcpy(elems_tile0 + (j_kpr772 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr772 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr773 = 0U;
            for (; i_kpr773 < 1U; i_kpr773++) {
                uint32_t j_kpr774 = i_kpr773;
                vec_memcpy(col_ind_tile0 + (j_kpr774 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr774 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 256U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 64U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr776 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr775 = 0U;
    for (; i_kpr775 < hi_kpr776; i_kpr775++) {
        uint32_t j_kpr777 = i_kpr775;
        elems_tile0[j_kpr777 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr777 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr777 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr777 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr778 = 0U;
    for (; i_kpr778 < 2U; i_kpr778++) {
        uint32_t j_kpr779 = i_kpr778;
        if (n_idx + j_kpr779 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr779 * 64U * 4U),
                out + j_kpr779 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x64x16
*/
static void
__hoisted_g_spmm_f32_512x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr780 = 0U;
        for (; i_kpr780 < 8U; i_kpr780++) {
            uint32_t j_kpr781 = i_kpr780;
            vec_memcpy(elems_tile0 + (j_kpr781 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr781 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr782 = 0U;
        for (; i_kpr782 < 8U; i_kpr782++) {
            uint32_t j_kpr783 = i_kpr782;
            vec_memcpy(col_ind_tile0 + (j_kpr783 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr783 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr784 = 0U;
        for (; i_kpr784 < to_; i_kpr784++)
            elems_tile0[i_kpr784 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr786 = 0U;
            for (; i_kpr786 < 8U; i_kpr786++) {
                uint32_t j_kpr787 = i_kpr786;
                vec_memcpy(elems_tile0 + (j_kpr787 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr787 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr788 = 0U;
            for (; i_kpr788 < 8U; i_kpr788++) {
                uint32_t j_kpr789 = i_kpr788;
                vec_memcpy(col_ind_tile0 + (j_kpr789 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr789 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr791 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr790 = 0U;
    for (; i_kpr790 < hi_kpr791; i_kpr790++) {
        uint32_t j_kpr792 = i_kpr790;
        elems_tile0[j_kpr792 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr792 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr792 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr792 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr793 = 0U;
    for (; i_kpr793 < 1U; i_kpr793++) {
        uint32_t j_kpr794 = i_kpr793;
        if (n_idx + j_kpr794 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr794 * 16U * 4U),
                out + j_kpr794 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x64x16_on
*/
static void
__hoisted_g_spmm_f32_512x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 64U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr795 = 0U;
        for (; i_kpr795 < 8U; i_kpr795++) {
            uint32_t j_kpr796 = i_kpr795;
            vec_memcpy(elems_tile0 + (j_kpr796 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr796 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr797 = 0U;
        for (; i_kpr797 < 8U; i_kpr797++) {
            uint32_t j_kpr798 = i_kpr797;
            vec_memcpy(col_ind_tile0 + (j_kpr798 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr798 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr799 = 0U;
        for (; i_kpr799 < to_; i_kpr799++)
            elems_tile0[i_kpr799 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
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
            uint32_t i_kpr801 = 0U;
            for (; i_kpr801 < 8U; i_kpr801++) {
                uint32_t j_kpr802 = i_kpr801;
                vec_memcpy(elems_tile0 + (j_kpr802 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr802 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr803 = 0U;
            for (; i_kpr803 < 8U; i_kpr803++) {
                uint32_t j_kpr804 = i_kpr803;
                vec_memcpy(col_ind_tile0 + (j_kpr804 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr804 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr806 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr805 = 0U;
    for (; i_kpr805 < hi_kpr806; i_kpr805++) {
        uint32_t j_kpr807 = i_kpr805;
        elems_tile0[j_kpr807 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr807 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr807 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr807 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr808 = 0U;
    for (; i_kpr808 < 1U; i_kpr808++) {
        uint32_t j_kpr809 = i_kpr808;
        if (n_idx + j_kpr809 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr809 * 16U * 4U),
                out + j_kpr809 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x128x16
*/
static void
__hoisted_g_spmm_f32_512x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr810 = 0U;
        for (; i_kpr810 < 8U; i_kpr810++) {
            uint32_t j_kpr811 = i_kpr810;
            vec_memcpy(elems_tile0 + (j_kpr811 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr811 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr812 = 0U;
        for (; i_kpr812 < 8U; i_kpr812++) {
            uint32_t j_kpr813 = i_kpr812;
            vec_memcpy(col_ind_tile0 + (j_kpr813 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr813 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr814 = 0U;
        for (; i_kpr814 < to_; i_kpr814++)
            elems_tile0[i_kpr814 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr816 = 0U;
            for (; i_kpr816 < 8U; i_kpr816++) {
                uint32_t j_kpr817 = i_kpr816;
                vec_memcpy(elems_tile0 + (j_kpr817 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr817 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr818 = 0U;
            for (; i_kpr818 < 8U; i_kpr818++) {
                uint32_t j_kpr819 = i_kpr818;
                vec_memcpy(col_ind_tile0 + (j_kpr819 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr819 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr821 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr820 = 0U;
    for (; i_kpr820 < hi_kpr821; i_kpr820++) {
        uint32_t j_kpr822 = i_kpr820;
        elems_tile0[j_kpr822 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr822 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr822 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr822 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr823 = 0U;
    for (; i_kpr823 < 2U; i_kpr823++) {
        uint32_t j_kpr824 = i_kpr823;
        if (n_idx + j_kpr824 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr824 * 16U * 4U),
                out + j_kpr824 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x128x16_on
*/
static void
__hoisted_g_spmm_f32_512x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr825 = 0U;
        for (; i_kpr825 < 8U; i_kpr825++) {
            uint32_t j_kpr826 = i_kpr825;
            vec_memcpy(elems_tile0 + (j_kpr826 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr826 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr827 = 0U;
        for (; i_kpr827 < 8U; i_kpr827++) {
            uint32_t j_kpr828 = i_kpr827;
            vec_memcpy(col_ind_tile0 + (j_kpr828 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr828 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr829 = 0U;
        for (; i_kpr829 < to_; i_kpr829++)
            elems_tile0[i_kpr829 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr831 = 0U;
            for (; i_kpr831 < 8U; i_kpr831++) {
                uint32_t j_kpr832 = i_kpr831;
                vec_memcpy(elems_tile0 + (j_kpr832 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr832 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr833 = 0U;
            for (; i_kpr833 < 8U; i_kpr833++) {
                uint32_t j_kpr834 = i_kpr833;
                vec_memcpy(col_ind_tile0 + (j_kpr834 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr834 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr836 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr835 = 0U;
    for (; i_kpr835 < hi_kpr836; i_kpr835++) {
        uint32_t j_kpr837 = i_kpr835;
        elems_tile0[j_kpr837 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr837 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr837 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr837 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr838 = 0U;
    for (; i_kpr838 < 2U; i_kpr838++) {
        uint32_t j_kpr839 = i_kpr838;
        if (n_idx + j_kpr839 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr839 * 16U * 4U),
                out + j_kpr839 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x128x32
*/
static void
__hoisted_g_spmm_f32_512x128x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr840 = 0U;
        for (; i_kpr840 < 4U; i_kpr840++) {
            uint32_t j_kpr841 = i_kpr840;
            vec_memcpy(elems_tile0 + (j_kpr841 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr841 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr842 = 0U;
        for (; i_kpr842 < 4U; i_kpr842++) {
            uint32_t j_kpr843 = i_kpr842;
            vec_memcpy(col_ind_tile0 + (j_kpr843 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr843 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr844 = 0U;
        for (; i_kpr844 < to_; i_kpr844++)
            elems_tile0[i_kpr844 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr846 = 0U;
            for (; i_kpr846 < 4U; i_kpr846++) {
                uint32_t j_kpr847 = i_kpr846;
                vec_memcpy(elems_tile0 + (j_kpr847 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr847 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr848 = 0U;
            for (; i_kpr848 < 4U; i_kpr848++) {
                uint32_t j_kpr849 = i_kpr848;
                vec_memcpy(col_ind_tile0 + (j_kpr849 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr849 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr851 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr850 = 0U;
    for (; i_kpr850 < hi_kpr851; i_kpr850++) {
        uint32_t j_kpr852 = i_kpr850;
        elems_tile0[j_kpr852 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr852 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr852 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr852 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr853 = 0U;
    for (; i_kpr853 < 1U; i_kpr853++) {
        uint32_t j_kpr854 = i_kpr853;
        if (n_idx + j_kpr854 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr854 * 32U * 4U),
                out + j_kpr854 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x128x32_on
*/
static void
__hoisted_g_spmm_f32_512x128x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 128U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr855 = 0U;
        for (; i_kpr855 < 4U; i_kpr855++) {
            uint32_t j_kpr856 = i_kpr855;
            vec_memcpy(elems_tile0 + (j_kpr856 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr856 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr857 = 0U;
        for (; i_kpr857 < 4U; i_kpr857++) {
            uint32_t j_kpr858 = i_kpr857;
            vec_memcpy(col_ind_tile0 + (j_kpr858 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr858 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr859 = 0U;
        for (; i_kpr859 < to_; i_kpr859++)
            elems_tile0[i_kpr859 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
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
            uint32_t i_kpr861 = 0U;
            for (; i_kpr861 < 4U; i_kpr861++) {
                uint32_t j_kpr862 = i_kpr861;
                vec_memcpy(elems_tile0 + (j_kpr862 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr862 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr863 = 0U;
            for (; i_kpr863 < 4U; i_kpr863++) {
                uint32_t j_kpr864 = i_kpr863;
                vec_memcpy(col_ind_tile0 + (j_kpr864 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr864 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr866 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr865 = 0U;
    for (; i_kpr865 < hi_kpr866; i_kpr865++) {
        uint32_t j_kpr867 = i_kpr865;
        elems_tile0[j_kpr867 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr867 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr867 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr867 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr868 = 0U;
    for (; i_kpr868 < 1U; i_kpr868++) {
        uint32_t j_kpr869 = i_kpr868;
        if (n_idx + j_kpr869 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr869 * 32U * 4U),
                out + j_kpr869 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x256x16
*/
static void
__hoisted_g_spmm_f32_512x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr870 = 0U;
        for (; i_kpr870 < 8U; i_kpr870++) {
            uint32_t j_kpr871 = i_kpr870;
            vec_memcpy(elems_tile0 + (j_kpr871 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr871 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr872 = 0U;
        for (; i_kpr872 < 8U; i_kpr872++) {
            uint32_t j_kpr873 = i_kpr872;
            vec_memcpy(col_ind_tile0 + (j_kpr873 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr873 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr874 = 0U;
        for (; i_kpr874 < to_; i_kpr874++)
            elems_tile0[i_kpr874 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr876 = 0U;
            for (; i_kpr876 < 8U; i_kpr876++) {
                uint32_t j_kpr877 = i_kpr876;
                vec_memcpy(elems_tile0 + (j_kpr877 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr877 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr878 = 0U;
            for (; i_kpr878 < 8U; i_kpr878++) {
                uint32_t j_kpr879 = i_kpr878;
                vec_memcpy(col_ind_tile0 + (j_kpr879 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr879 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr881 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr880 = 0U;
    for (; i_kpr880 < hi_kpr881; i_kpr880++) {
        uint32_t j_kpr882 = i_kpr880;
        elems_tile0[j_kpr882 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr882 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr882 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr882 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr883 = 0U;
    for (; i_kpr883 < 4U; i_kpr883++) {
        uint32_t j_kpr884 = i_kpr883;
        if (n_idx + j_kpr884 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr884 * 16U * 4U),
                out + j_kpr884 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x256x16_on
*/
static void
__hoisted_g_spmm_f32_512x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr885 = 0U;
        for (; i_kpr885 < 8U; i_kpr885++) {
            uint32_t j_kpr886 = i_kpr885;
            vec_memcpy(elems_tile0 + (j_kpr886 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr886 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr887 = 0U;
        for (; i_kpr887 < 8U; i_kpr887++) {
            uint32_t j_kpr888 = i_kpr887;
            vec_memcpy(col_ind_tile0 + (j_kpr888 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr888 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr889 = 0U;
        for (; i_kpr889 < to_; i_kpr889++)
            elems_tile0[i_kpr889 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr891 = 0U;
            for (; i_kpr891 < 8U; i_kpr891++) {
                uint32_t j_kpr892 = i_kpr891;
                vec_memcpy(elems_tile0 + (j_kpr892 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr892 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr893 = 0U;
            for (; i_kpr893 < 8U; i_kpr893++) {
                uint32_t j_kpr894 = i_kpr893;
                vec_memcpy(col_ind_tile0 + (j_kpr894 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr894 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr896 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr895 = 0U;
    for (; i_kpr895 < hi_kpr896; i_kpr895++) {
        uint32_t j_kpr897 = i_kpr895;
        elems_tile0[j_kpr897 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr897 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr897 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr897 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr898 = 0U;
    for (; i_kpr898 < 4U; i_kpr898++) {
        uint32_t j_kpr899 = i_kpr898;
        if (n_idx + j_kpr899 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr899 * 16U * 4U),
                out + j_kpr899 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x256x32
*/
static void
__hoisted_g_spmm_f32_512x256x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr900 = 0U;
        for (; i_kpr900 < 4U; i_kpr900++) {
            uint32_t j_kpr901 = i_kpr900;
            vec_memcpy(elems_tile0 + (j_kpr901 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr901 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr902 = 0U;
        for (; i_kpr902 < 4U; i_kpr902++) {
            uint32_t j_kpr903 = i_kpr902;
            vec_memcpy(col_ind_tile0 + (j_kpr903 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr903 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr904 = 0U;
        for (; i_kpr904 < to_; i_kpr904++)
            elems_tile0[i_kpr904 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr906 = 0U;
            for (; i_kpr906 < 4U; i_kpr906++) {
                uint32_t j_kpr907 = i_kpr906;
                vec_memcpy(elems_tile0 + (j_kpr907 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr907 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr908 = 0U;
            for (; i_kpr908 < 4U; i_kpr908++) {
                uint32_t j_kpr909 = i_kpr908;
                vec_memcpy(col_ind_tile0 + (j_kpr909 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr909 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr911 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr910 = 0U;
    for (; i_kpr910 < hi_kpr911; i_kpr910++) {
        uint32_t j_kpr912 = i_kpr910;
        elems_tile0[j_kpr912 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr912 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr912 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr912 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr913 = 0U;
    for (; i_kpr913 < 2U; i_kpr913++) {
        uint32_t j_kpr914 = i_kpr913;
        if (n_idx + j_kpr914 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr914 * 32U * 4U),
                out + j_kpr914 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x256x32_on
*/
static void
__hoisted_g_spmm_f32_512x256x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr915 = 0U;
        for (; i_kpr915 < 4U; i_kpr915++) {
            uint32_t j_kpr916 = i_kpr915;
            vec_memcpy(elems_tile0 + (j_kpr916 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr916 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr917 = 0U;
        for (; i_kpr917 < 4U; i_kpr917++) {
            uint32_t j_kpr918 = i_kpr917;
            vec_memcpy(col_ind_tile0 + (j_kpr918 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr918 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr919 = 0U;
        for (; i_kpr919 < to_; i_kpr919++)
            elems_tile0[i_kpr919 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr921 = 0U;
            for (; i_kpr921 < 4U; i_kpr921++) {
                uint32_t j_kpr922 = i_kpr921;
                vec_memcpy(elems_tile0 + (j_kpr922 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr922 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr923 = 0U;
            for (; i_kpr923 < 4U; i_kpr923++) {
                uint32_t j_kpr924 = i_kpr923;
                vec_memcpy(col_ind_tile0 + (j_kpr924 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr924 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr926 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr925 = 0U;
    for (; i_kpr925 < hi_kpr926; i_kpr925++) {
        uint32_t j_kpr927 = i_kpr925;
        elems_tile0[j_kpr927 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr927 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr927 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr927 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr928 = 0U;
    for (; i_kpr928 < 2U; i_kpr928++) {
        uint32_t j_kpr929 = i_kpr928;
        if (n_idx + j_kpr929 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr929 * 32U * 4U),
                out + j_kpr929 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x256x64
*/
static void
__hoisted_g_spmm_f32_512x256x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr930 = 0U;
        for (; i_kpr930 < 2U; i_kpr930++) {
            uint32_t j_kpr931 = i_kpr930;
            vec_memcpy(elems_tile0 + (j_kpr931 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr931 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr932 = 0U;
        for (; i_kpr932 < 2U; i_kpr932++) {
            uint32_t j_kpr933 = i_kpr932;
            vec_memcpy(col_ind_tile0 + (j_kpr933 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr933 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr934 = 0U;
        for (; i_kpr934 < to_; i_kpr934++)
            elems_tile0[i_kpr934 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
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
            uint32_t i_kpr936 = 0U;
            for (; i_kpr936 < 2U; i_kpr936++) {
                uint32_t j_kpr937 = i_kpr936;
                vec_memcpy(elems_tile0 + (j_kpr937 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr937 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr938 = 0U;
            for (; i_kpr938 < 2U; i_kpr938++) {
                uint32_t j_kpr939 = i_kpr938;
                vec_memcpy(col_ind_tile0 + (j_kpr939 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr939 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr941 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr940 = 0U;
    for (; i_kpr940 < hi_kpr941; i_kpr940++) {
        uint32_t j_kpr942 = i_kpr940;
        elems_tile0[j_kpr942 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr942 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr942 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr942 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr943 = 0U;
    for (; i_kpr943 < 1U; i_kpr943++) {
        uint32_t j_kpr944 = i_kpr943;
        if (n_idx + j_kpr944 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr944 * 64U * 4U),
                out + j_kpr944 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x256x64_on
*/
static void
__hoisted_g_spmm_f32_512x256x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 256U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr945 = 0U;
        for (; i_kpr945 < 2U; i_kpr945++) {
            uint32_t j_kpr946 = i_kpr945;
            vec_memcpy(elems_tile0 + (j_kpr946 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr946 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr947 = 0U;
        for (; i_kpr947 < 2U; i_kpr947++) {
            uint32_t j_kpr948 = i_kpr947;
            vec_memcpy(col_ind_tile0 + (j_kpr948 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr948 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr949 = 0U;
        for (; i_kpr949 < to_; i_kpr949++)
            elems_tile0[i_kpr949 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
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
            uint32_t i_kpr951 = 0U;
            for (; i_kpr951 < 2U; i_kpr951++) {
                uint32_t j_kpr952 = i_kpr951;
                vec_memcpy(elems_tile0 + (j_kpr952 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr952 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr953 = 0U;
            for (; i_kpr953 < 2U; i_kpr953++) {
                uint32_t j_kpr954 = i_kpr953;
                vec_memcpy(col_ind_tile0 + (j_kpr954 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr954 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr956 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr955 = 0U;
    for (; i_kpr955 < hi_kpr956; i_kpr955++) {
        uint32_t j_kpr957 = i_kpr955;
        elems_tile0[j_kpr957 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr957 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr957 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr957 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr958 = 0U;
    for (; i_kpr958 < 1U; i_kpr958++) {
        uint32_t j_kpr959 = i_kpr958;
        if (n_idx + j_kpr959 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr959 * 64U * 4U),
                out + j_kpr959 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x512x16
*/
static void
__hoisted_g_spmm_f32_512x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr960 = 0U;
        for (; i_kpr960 < 8U; i_kpr960++) {
            uint32_t j_kpr961 = i_kpr960;
            vec_memcpy(elems_tile0 + (j_kpr961 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr961 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr962 = 0U;
        for (; i_kpr962 < 8U; i_kpr962++) {
            uint32_t j_kpr963 = i_kpr962;
            vec_memcpy(col_ind_tile0 + (j_kpr963 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr963 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr964 = 0U;
        for (; i_kpr964 < to_; i_kpr964++)
            elems_tile0[i_kpr964 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr966 = 0U;
            for (; i_kpr966 < 8U; i_kpr966++) {
                uint32_t j_kpr967 = i_kpr966;
                vec_memcpy(elems_tile0 + (j_kpr967 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr967 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr968 = 0U;
            for (; i_kpr968 < 8U; i_kpr968++) {
                uint32_t j_kpr969 = i_kpr968;
                vec_memcpy(col_ind_tile0 + (j_kpr969 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr969 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr971 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr970 = 0U;
    for (; i_kpr970 < hi_kpr971; i_kpr970++) {
        uint32_t j_kpr972 = i_kpr970;
        elems_tile0[j_kpr972 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr972 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr972 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr972 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr973 = 0U;
    for (; i_kpr973 < 8U; i_kpr973++) {
        uint32_t j_kpr974 = i_kpr973;
        if (n_idx + j_kpr974 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr974 * 16U * 4U),
                out + j_kpr974 * 4U);
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x512x16_on
*/
static void
__hoisted_g_spmm_f32_512x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[32U];
    memset(out, 0U, 32U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr975 = 0U;
        for (; i_kpr975 < 8U; i_kpr975++) {
            uint32_t j_kpr976 = i_kpr975;
            vec_memcpy(elems_tile0 + (j_kpr976 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr976 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr977 = 0U;
        for (; i_kpr977 < 8U; i_kpr977++) {
            uint32_t j_kpr978 = i_kpr977;
            vec_memcpy(col_ind_tile0 + (j_kpr978 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr978 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr979 = 0U;
        for (; i_kpr979 < to_; i_kpr979++)
            elems_tile0[i_kpr979 * 16U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 8U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 16U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 16U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr981 = 0U;
            for (; i_kpr981 < 8U; i_kpr981++) {
                uint32_t j_kpr982 = i_kpr981;
                vec_memcpy(elems_tile0 + (j_kpr982 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr982 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr983 = 0U;
            for (; i_kpr983 < 8U; i_kpr983++) {
                uint32_t j_kpr984 = i_kpr983;
                vec_memcpy(col_ind_tile0 + (j_kpr984 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr984 * 16U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 8U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 16U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 16U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr986 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr985 = 0U;
    for (; i_kpr985 < hi_kpr986; i_kpr985++) {
        uint32_t j_kpr987 = i_kpr985;
        elems_tile0[j_kpr987 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr987 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr987 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr987 * 16U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 16U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr988 = 0U;
    for (; i_kpr988 < 8U; i_kpr988++) {
        uint32_t j_kpr989 = i_kpr988;
        if (n_idx + j_kpr989 * 16U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr989 * 16U * 4U),
                out + j_kpr989 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x512x32
*/
static void
__hoisted_g_spmm_f32_512x512x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr990 = 0U;
        for (; i_kpr990 < 4U; i_kpr990++) {
            uint32_t j_kpr991 = i_kpr990;
            vec_memcpy(elems_tile0 + (j_kpr991 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr991 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr992 = 0U;
        for (; i_kpr992 < 4U; i_kpr992++) {
            uint32_t j_kpr993 = i_kpr992;
            vec_memcpy(col_ind_tile0 + (j_kpr993 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr993 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr994 = 0U;
        for (; i_kpr994 < to_; i_kpr994++)
            elems_tile0[i_kpr994 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr996 = 0U;
            for (; i_kpr996 < 4U; i_kpr996++) {
                uint32_t j_kpr997 = i_kpr996;
                vec_memcpy(elems_tile0 + (j_kpr997 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr997 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr998 = 0U;
            for (; i_kpr998 < 4U; i_kpr998++) {
                uint32_t j_kpr999 = i_kpr998;
                vec_memcpy(col_ind_tile0 + (j_kpr999 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr999 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr1001 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1000 = 0U;
    for (; i_kpr1000 < hi_kpr1001; i_kpr1000++) {
        uint32_t j_kpr1002 = i_kpr1000;
        elems_tile0[j_kpr1002 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1002 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1002 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1002 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr1003 = 0U;
    for (; i_kpr1003 < 4U; i_kpr1003++) {
        uint32_t j_kpr1004 = i_kpr1003;
        if (n_idx + j_kpr1004 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr1004 * 32U * 4U),
                out + j_kpr1004 * 4U);
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x512x32_on
*/
static void
__hoisted_g_spmm_f32_512x512x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[16U];
    memset(out, 0U, 16U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr1005 = 0U;
        for (; i_kpr1005 < 4U; i_kpr1005++) {
            uint32_t j_kpr1006 = i_kpr1005;
            vec_memcpy(elems_tile0 + (j_kpr1006 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1006 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1007 = 0U;
        for (; i_kpr1007 < 4U; i_kpr1007++) {
            uint32_t j_kpr1008 = i_kpr1007;
            vec_memcpy(col_ind_tile0 + (j_kpr1008 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1008 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1009 = 0U;
        for (; i_kpr1009 < to_; i_kpr1009++)
            elems_tile0[i_kpr1009 * 32U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 4U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 32U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 32U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1011 = 0U;
            for (; i_kpr1011 < 4U; i_kpr1011++) {
                uint32_t j_kpr1012 = i_kpr1011;
                vec_memcpy(elems_tile0 + (j_kpr1012 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1012 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1013 = 0U;
            for (; i_kpr1013 < 4U; i_kpr1013++) {
                uint32_t j_kpr1014 = i_kpr1013;
                vec_memcpy(col_ind_tile0 + (j_kpr1014 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1014 * 32U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 4U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 32U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 32U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr1016 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1015 = 0U;
    for (; i_kpr1015 < hi_kpr1016; i_kpr1015++) {
        uint32_t j_kpr1017 = i_kpr1015;
        elems_tile0[j_kpr1017 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1017 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1017 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1017 * 32U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 32U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr1018 = 0U;
    for (; i_kpr1018 < 4U; i_kpr1018++) {
        uint32_t j_kpr1019 = i_kpr1018;
        if (n_idx + j_kpr1019 * 32U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr1019 * 32U * 4U),
                out + j_kpr1019 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x512x64
*/
static void
__hoisted_g_spmm_f32_512x512x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr1020 = 0U;
        for (; i_kpr1020 < 2U; i_kpr1020++) {
            uint32_t j_kpr1021 = i_kpr1020;
            vec_memcpy(elems_tile0 + (j_kpr1021 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1021 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1022 = 0U;
        for (; i_kpr1022 < 2U; i_kpr1022++) {
            uint32_t j_kpr1023 = i_kpr1022;
            vec_memcpy(col_ind_tile0 + (j_kpr1023 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1023 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1024 = 0U;
        for (; i_kpr1024 < to_; i_kpr1024++)
            elems_tile0[i_kpr1024 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1026 = 0U;
            for (; i_kpr1026 < 2U; i_kpr1026++) {
                uint32_t j_kpr1027 = i_kpr1026;
                vec_memcpy(elems_tile0 + (j_kpr1027 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1027 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1028 = 0U;
            for (; i_kpr1028 < 2U; i_kpr1028++) {
                uint32_t j_kpr1029 = i_kpr1028;
                vec_memcpy(col_ind_tile0 + (j_kpr1029 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1029 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 64U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr1031 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1030 = 0U;
    for (; i_kpr1030 < hi_kpr1031; i_kpr1030++) {
        uint32_t j_kpr1032 = i_kpr1030;
        elems_tile0[j_kpr1032 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1032 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1032 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1032 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr1033 = 0U;
    for (; i_kpr1033 < 2U; i_kpr1033++) {
        uint32_t j_kpr1034 = i_kpr1033;
        if (n_idx + j_kpr1034 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr1034 * 64U * 4U),
                out + j_kpr1034 * 4U);
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x512x64_on
*/
static void
__hoisted_g_spmm_f32_512x512x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr1035 = 0U;
        for (; i_kpr1035 < 2U; i_kpr1035++) {
            uint32_t j_kpr1036 = i_kpr1035;
            vec_memcpy(elems_tile0 + (j_kpr1036 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1036 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1037 = 0U;
        for (; i_kpr1037 < 2U; i_kpr1037++) {
            uint32_t j_kpr1038 = i_kpr1037;
            vec_memcpy(col_ind_tile0 + (j_kpr1038 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1038 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1039 = 0U;
        for (; i_kpr1039 < to_; i_kpr1039++)
            elems_tile0[i_kpr1039 * 64U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 64U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf01 * 64U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                        }
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1041 = 0U;
            for (; i_kpr1041 < 2U; i_kpr1041++) {
                uint32_t j_kpr1042 = i_kpr1041;
                vec_memcpy(elems_tile0 + (j_kpr1042 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1042 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1043 = 0U;
            for (; i_kpr1043 < 2U; i_kpr1043++) {
                uint32_t j_kpr1044 = i_kpr1043;
                vec_memcpy(col_ind_tile0 + (j_kpr1044 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1044 * 64U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 64U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + n_idx + __anf03 * 64U * 4U));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t ixv = ix;
                                out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                            }
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr1046 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1045 = 0U;
    for (; i_kpr1045 < hi_kpr1046; i_kpr1045++) {
        uint32_t j_kpr1047 = i_kpr1045;
        elems_tile0[j_kpr1047 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1047 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1047 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1047 * 64U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf02 * 64U * 4U));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t ixv = ix;
                        out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                    }
                }
            }
        }
    }
    uint32_t i_kpr1048 = 0U;
    for (; i_kpr1048 < 2U; i_kpr1048++) {
        uint32_t j_kpr1049 = i_kpr1048;
        if (n_idx + j_kpr1049 * 64U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr1049 * 64U * 4U),
                out + j_kpr1049 * 4U);
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_spmm_f32_512x512x128
*/
static void
__hoisted_g_spmm_f32_512x512x128_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr1050 = 0U;
        for (; i_kpr1050 < 1U; i_kpr1050++) {
            uint32_t j_kpr1051 = i_kpr1050;
            vec_memcpy(elems_tile0 + (j_kpr1051 * 128U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1051 * 128U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1052 = 0U;
        for (; i_kpr1052 < 1U; i_kpr1052++) {
            uint32_t j_kpr1053 = i_kpr1052;
            vec_memcpy(col_ind_tile0 + (j_kpr1053 * 128U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1053 * 128U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i_kpr1054 = 0U;
        for (; i_kpr1054 < to_; i_kpr1054++)
            elems_tile0[i_kpr1054 * 128U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 128U * 4U));
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
            uint32_t i_kpr1056 = 0U;
            for (; i_kpr1056 < 1U; i_kpr1056++) {
                uint32_t j_kpr1057 = i_kpr1056;
                vec_memcpy(elems_tile0 + (j_kpr1057 * 128U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1057 * 128U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1058 = 0U;
            for (; i_kpr1058 < 1U; i_kpr1058++) {
                uint32_t j_kpr1059 = i_kpr1058;
                vec_memcpy(
                    col_ind_tile0 + (j_kpr1059 * 128U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1059 * 128U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 128U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr1061 = (__anf01 + 127U - threadIdx.x) / 128U;
    uint32_t i_kpr1060 = 0U;
    for (; i_kpr1060 < hi_kpr1061; i_kpr1060++) {
        uint32_t j_kpr1062 = i_kpr1060;
        elems_tile0[j_kpr1062 * 128U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1062 * 128U + threadIdx.x];
        col_ind_tile0[j_kpr1062 * 128U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1062 * 128U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 128U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr1063 = 0U;
    for (; i_kpr1063 < 1U; i_kpr1063++) {
        uint32_t j_kpr1064 = i_kpr1063;
        if (n_idx + j_kpr1064 * 128U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr1064 * 128U * 4U),
                out + j_kpr1064 * 4U);
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_spmm_f32_512x512x128_on
*/
static void
__hoisted_g_spmm_f32_512x512x128_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t cols, float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 512U + threadIdx.x * 4U;
    float *elems_tile0 = (float *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 512U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 4U * 4U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[4U];
    memset(out, 0U, 4U * sizeof(float));
    if (nnz >= 512U) {
        uint32_t i_kpr1065 = 0U;
        for (; i_kpr1065 < 1U; i_kpr1065++) {
            uint32_t j_kpr1066 = i_kpr1065;
            vec_memcpy(elems_tile0 + (j_kpr1066 * 128U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1066 * 128U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1067 = 0U;
        for (; i_kpr1067 < 1U; i_kpr1067++) {
            uint32_t j_kpr1068 = i_kpr1067;
            vec_memcpy(col_ind_tile0 + (j_kpr1068 * 128U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1068 * 128U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i_kpr1069 = 0U;
        for (; i_kpr1069 < to_; i_kpr1069++)
            elems_tile0[i_kpr1069 * 128U + threadIdx.x] = 0.0f;
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 512U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                float kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + n_idx + __anf01 * 128U * 4U));
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
            uint32_t i_kpr1071 = 0U;
            for (; i_kpr1071 < 1U; i_kpr1071++) {
                uint32_t j_kpr1072 = i_kpr1071;
                vec_memcpy(elems_tile0 + (j_kpr1072 * 128U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1072 * 128U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1073 = 0U;
            for (; i_kpr1073 < 1U; i_kpr1073++) {
                uint32_t j_kpr1074 = i_kpr1073;
                vec_memcpy(
                    col_ind_tile0 + (j_kpr1074 * 128U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1074 * 128U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 512U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    float kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + n_idx + __anf03 * 128U * 4U));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t ixv = ix;
                            out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
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
    uint32_t __anf01 = nnz;
    __syncthreads();
    uint32_t hi_kpr1076 = (__anf01 + 127U - threadIdx.x) / 128U;
    uint32_t i_kpr1075 = 0U;
    for (; i_kpr1075 < hi_kpr1076; i_kpr1075++) {
        uint32_t j_kpr1077 = i_kpr1075;
        elems_tile0[j_kpr1077 * 128U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1077 * 128U + threadIdx.x];
        col_ind_tile0[j_kpr1077 * 128U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1077 * 128U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            float kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + n_idx + __anf02 * 128U * 4U));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++) {
                    uint32_t ixv = ix;
                    out[__anf1 * 4U + ixv] += kx * lchunk[ixv];
                }
            }
        }
    }
    uint32_t i_kpr1078 = 0U;
    for (; i_kpr1078 < 1U; i_kpr1078++) {
        uint32_t j_kpr1079 = i_kpr1078;
        if (n_idx + j_kpr1079 * 128U * 4U < cols)
            vec_memcpy(gC + (cols * m_idx + n_idx + j_kpr1079 * 128U * 4U),
                out + j_kpr1079 * 4U);
    }
}

void Klas_SPMM_spmm_u32(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__uint32_t gA, uint32_t *row_indices,
    uint32_t *gB, uint32_t *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * 128U +
                   (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(uint32_t) * 128U +
            (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_spmm_u32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_spmm_u32_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(uint32_t) * 128U + (uint32_t) sizeof(uint32_t) * 128U,
        s, row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_spmm_f32(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_spmm_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_spmm_f32_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_32x4x1(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x4x1_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x4x1_0,
        rows * (cols / 4U + (uint32_t) (cols % 4U != 0U)), 1U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_32x8x2(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x8x2_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x8x2_0,
        rows * (cols / 8U + (uint32_t) (cols % 8U != 0U)), 2U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_32x16x4(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x16x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x16x4_0,
        rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)), 4U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_32x32x8(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x32x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x32x8_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 8U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_32x64x8(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x64x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x64x8_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 8U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_32x4x1_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x4x1_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x4x1_on_0,
        rows * (cols / 4U + (uint32_t) (cols % 4U != 0U)), 1U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_32x8x2_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x8x2_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x8x2_on_0,
        rows * (cols / 8U + (uint32_t) (cols % 8U != 0U)), 2U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_32x16x4_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x16x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x16x4_on_0,
        rows * (cols / 16U + (uint32_t) (cols % 16U != 0U)), 4U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_32x32x8_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x32x8_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x32x8_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 8U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_32x64x8_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_32x64x8_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f32_32x64x8_on_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 8U,
        (uint32_t) sizeof(float) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_64x64x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x64x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x16_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_64x64x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x64x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x64x16_on_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_64x128x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x128x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x16_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_64x128x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x128x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x128x16_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_64x256x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x256x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x16_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_64x256x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x256x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x256x16_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_64x512x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x512x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x16_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_64x512x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U);
    if ((uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_64x512x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 64U +
                (uint32_t) sizeof(uint32_t) * 64U));
    KPR_KCALL(__hoisted_g_spmm_f32_64x512x16_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 64U + (uint32_t) sizeof(uint32_t) * 64U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x64x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x64x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x16_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x64x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x64x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x64x16_on_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x128x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x16_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x128x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x16_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x128x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x32_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x128x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x128x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x128x32_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x256x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x16_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x256x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x16_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x256x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x32_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x256x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x256x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x256x32_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x512x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x16_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x512x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x16_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_128x512x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x32_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_128x512x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U);
    if ((uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_128x512x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 128U +
                (uint32_t) sizeof(uint32_t) * 128U));
    KPR_KCALL(__hoisted_g_spmm_f32_128x512x32_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U,
        (uint32_t) sizeof(float) * 128U + (uint32_t) sizeof(uint32_t) * 128U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x64x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x64x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x16_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x64x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x64x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x64x16_on_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x128x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x16_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x128x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x16_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x128x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x32_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x128x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x128x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x128x32_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x256x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x16_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x256x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x16_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x256x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x32_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x256x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x32_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x256x64(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x64_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x256x64_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x256x64_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x256x64_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x512x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x16_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x512x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x16_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x512x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x32_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x512x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x32_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_256x512x64(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x64_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_256x512x64_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_256x512x64_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_spmm_f32_256x512x64_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(uint32_t) * 256U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x64x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x64x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x16_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x64x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x64x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x64x16_on_0,
        rows * (cols / 64U + (uint32_t) (cols % 64U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x128x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x16_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x128x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x16_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x128x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x32_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x128x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x128x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x128x32_on_0,
        rows * (cols / 128U + (uint32_t) (cols % 128U != 0U)), 32U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x256x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x16_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x256x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x16_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x256x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x32_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x256x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x32_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 32U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x256x64(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x64_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x256x64_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x256x64_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x256x64_on_0,
        rows * (cols / 256U + (uint32_t) (cols % 256U != 0U)), 64U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x512x16(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x16_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x512x16_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x16_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x16_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 16U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x512x32(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x32_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x512x32_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x32_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x32_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 32U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x512x64(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x64_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x512x64_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x64_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x64_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 64U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f32_512x512x128(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x128_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x128_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 128U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f32_512x512x128_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_smatrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KPR_GUARD(rows < 10000U);
    KPR_GUARD(shared < 10000U);
    KPR_GUARD(cols < 10000U);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U);
    if ((uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f32_512x512x128_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 512U +
                (uint32_t) sizeof(uint32_t) * 512U));
    KPR_KCALL(__hoisted_g_spmm_f32_512x512x128_on_0,
        rows * (cols / 512U + (uint32_t) (cols % 512U != 0U)), 128U,
        (uint32_t) sizeof(float) * 512U + (uint32_t) sizeof(uint32_t) * 512U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_spmm_f32_dispatch(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_smatrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    if (cols % 64U == 0U)
        Klas_SPMM_g_spmm_f32_32x64x8(
            rows, shared, cols, gA, row_indices, gB, gC);
    else if (cols % 32U == 0U)
        Klas_SPMM_g_spmm_f32_32x32x8(
            rows, shared, cols, gA, row_indices, gB, gC);
    else if (cols % 16U == 0U)
        Klas_SPMM_g_spmm_f32_32x16x4(
            rows, shared, cols, gA, row_indices, gB, gC);
    else if (cols % 8U == 0U)
        Klas_SPMM_g_spmm_f32_32x8x2(
            rows, shared, cols, gA, row_indices, gB, gC);
    else
        Klas_SPMM_g_spmm_f32_32x4x1(
            rows, shared, cols, gA, row_indices, gB, gC);
}
