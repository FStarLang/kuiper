
#include "Klas_SPMM.h"

__global__ __launch_bounds__(32)
/**
  hoisted when extracting spmm_u32
*/
static void
__hoisted_spmm_u32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__uint32_t gA, uint32_t cols, uint32_t *gB,
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr13 = 0U;
    for (; i_kpr13 < 1U; i_kpr13++) {
        uint32_t j_kpr14 = i_kpr13;
        uint32_t off = n_idx + j_kpr14 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            uint32_t combined[4U] = {0U};
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr14 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting spmm_f32
*/
static void
__hoisted_spmm_f32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr28 = 0U;
    for (; i_kpr28 < 2U; i_kpr28++) {
        uint32_t j_kpr29 = i_kpr28;
        uint32_t off = n_idx + j_kpr29 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr29 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_f16_f32_f32_32x32x4
*/
static void
__hoisted_g_spmm_f16_f16_f32_f32_32x32x4_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols, half *gB,
    float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr30 = 0U;
        for (; i_kpr30 < 1U; i_kpr30++) {
            uint32_t j_kpr31 = i_kpr30;
            vec_memcpy(elems_tile0 + (j_kpr31 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr31 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr32 = 0U;
        for (; i_kpr32 < 2U; i_kpr32++) {
            uint32_t j_kpr33 = i_kpr32;
            vec_memcpy(col_ind_tile0 + (j_kpr33 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr33 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr34 = 0U;
        for (; i_kpr34 < to_; i_kpr34++) {
            uint32_t j_kpr35 = i_kpr34;
            elems_tile0[j_kpr35 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        half lchunk[4U];
                        for (uint32_t _i = 0U; _i < 4U; ++_i)
                            lchunk[_i] = __float2half_rn(0.0f);
                        uint32_t lane = 0U;
                        for (; lane < 4U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf01 * 4U * 4U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 4U + ix];
                            half __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 +
                                __half2float(kx) * __half2float(__anf11);
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
            uint32_t i_kpr36 = 0U;
            for (; i_kpr36 < 1U; i_kpr36++) {
                uint32_t j_kpr37 = i_kpr36;
                vec_memcpy(elems_tile0 + (j_kpr37 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr37 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr38 = 0U;
            for (; i_kpr38 < 2U; i_kpr38++) {
                uint32_t j_kpr39 = i_kpr38;
                vec_memcpy(col_ind_tile0 + (j_kpr39 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr39 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            half lchunk[4U];
                            for (uint32_t _i = 0U; _i < 4U; ++_i)
                                lchunk[_i] = __float2half_rn(0.0f);
                            uint32_t lane = 0U;
                            for (; lane < 4U; lane++)
                                lchunk[lane] =
                                    gB[kr * cols +
                                        (n_idx + __anf03 * 4U * 4U + lane)];
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                float __anf3 = out[__anf1 * 4U + ix];
                                half __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 +
                                    __half2float(kx) * __half2float(__anf11);
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
    uint32_t hi_kpr41 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr40 = 0U;
    for (; i_kpr40 < hi_kpr41; i_kpr40++) {
        uint32_t j_kpr42 = i_kpr40;
        elems_tile0[j_kpr42 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr42 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr42 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr42 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    half lchunk[4U];
                    for (uint32_t _i = 0U; _i < 4U; ++_i)
                        lchunk[_i] = __float2half_rn(0.0f);
                    uint32_t lane = 0U;
                    for (; lane < 4U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf02 * 4U * 4U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 4U + ix];
                        half __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 + __half2float(kx) * __half2float(__anf11);
                    }
                }
            }
        }
    }
    uint32_t i_kpr43 = 0U;
    for (; i_kpr43 < 2U; i_kpr43++) {
        uint32_t j_kpr44 = i_kpr43;
        uint32_t off = n_idx + j_kpr44 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr44 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_f16_f32_f32_32x32x4_on
*/
static void
__hoisted_g_spmm_f16_f16_f32_f32_32x32x4_on_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    half *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr45 = 0U;
        for (; i_kpr45 < 1U; i_kpr45++) {
            uint32_t j_kpr46 = i_kpr45;
            vec_memcpy(elems_tile0 + (j_kpr46 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr46 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr47 = 0U;
        for (; i_kpr47 < 2U; i_kpr47++) {
            uint32_t j_kpr48 = i_kpr47;
            vec_memcpy(col_ind_tile0 + (j_kpr48 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr48 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr49 = 0U;
        for (; i_kpr49 < to_; i_kpr49++) {
            uint32_t j_kpr50 = i_kpr49;
            elems_tile0[j_kpr50 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        half lchunk[4U];
                        for (uint32_t _i = 0U; _i < 4U; ++_i)
                            lchunk[_i] = __float2half_rn(0.0f);
                        uint32_t lane = 0U;
                        for (; lane < 4U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf01 * 4U * 4U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 4U + ix];
                            half __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 +
                                __half2float(kx) * __half2float(__anf11);
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
            uint32_t i_kpr51 = 0U;
            for (; i_kpr51 < 1U; i_kpr51++) {
                uint32_t j_kpr52 = i_kpr51;
                vec_memcpy(elems_tile0 + (j_kpr52 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr52 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr53 = 0U;
            for (; i_kpr53 < 2U; i_kpr53++) {
                uint32_t j_kpr54 = i_kpr53;
                vec_memcpy(col_ind_tile0 + (j_kpr54 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr54 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            half lchunk[4U];
                            for (uint32_t _i = 0U; _i < 4U; ++_i)
                                lchunk[_i] = __float2half_rn(0.0f);
                            uint32_t lane = 0U;
                            for (; lane < 4U; lane++)
                                lchunk[lane] =
                                    gB[kr * cols +
                                        (n_idx + __anf03 * 4U * 4U + lane)];
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                float __anf3 = out[__anf1 * 4U + ix];
                                half __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 +
                                    __half2float(kx) * __half2float(__anf11);
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
    uint32_t hi_kpr56 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr55 = 0U;
    for (; i_kpr55 < hi_kpr56; i_kpr55++) {
        uint32_t j_kpr57 = i_kpr55;
        elems_tile0[j_kpr57 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr57 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr57 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr57 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    half lchunk[4U];
                    for (uint32_t _i = 0U; _i < 4U; ++_i)
                        lchunk[_i] = __float2half_rn(0.0f);
                    uint32_t lane = 0U;
                    for (; lane < 4U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf02 * 4U * 4U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 4U + ix];
                        half __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 + __half2float(kx) * __half2float(__anf11);
                    }
                }
            }
        }
    }
    uint32_t i_kpr58 = 0U;
    for (; i_kpr58 < 2U; i_kpr58++) {
        uint32_t j_kpr59 = i_kpr58;
        uint32_t off = n_idx + j_kpr59 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr59 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_bf16_bf16_f32_f32_32x32x4
*/
static void
__hoisted_g_spmm_bf16_bf16_f32_f32_32x32x4_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix____nv_bfloat16 gA,
    uint32_t cols, __nv_bfloat16 *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    __nv_bfloat16 *elems_tile0 = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(__nv_bfloat16) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr60 = 0U;
        for (; i_kpr60 < 1U; i_kpr60++) {
            uint32_t j_kpr61 = i_kpr60;
            vec_memcpy(elems_tile0 + (j_kpr61 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr61 * 4U + threadIdx.x) * 8U));
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
        for (; i_kpr64 < to_; i_kpr64++) {
            uint32_t j_kpr65 = i_kpr64;
            elems_tile0[j_kpr65 * 4U + threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                __nv_bfloat16 kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        __nv_bfloat16 lchunk[4U];
                        for (uint32_t _i = 0U; _i < 4U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t lane = 0U;
                        for (; lane < 4U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf01 * 4U * 4U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 4U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 + __bfloat162float(kx) *
                                             __bfloat162float(__anf11);
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
            uint32_t i_kpr66 = 0U;
            for (; i_kpr66 < 1U; i_kpr66++) {
                uint32_t j_kpr67 = i_kpr66;
                vec_memcpy(elems_tile0 + (j_kpr67 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr67 * 4U + threadIdx.x) * 8U));
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
                    __nv_bfloat16 kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            __nv_bfloat16 lchunk[4U];
                            for (uint32_t _i = 0U; _i < 4U; ++_i)
                                lchunk[_i] = __float2bfloat16(0.0f);
                            uint32_t lane = 0U;
                            for (; lane < 4U; lane++)
                                lchunk[lane] =
                                    gB[kr * cols +
                                        (n_idx + __anf03 * 4U * 4U + lane)];
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                float __anf3 = out[__anf1 * 4U + ix];
                                __nv_bfloat16 __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 + __bfloat162float(kx) *
                                                 __bfloat162float(__anf11);
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
            __nv_bfloat16 kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    __nv_bfloat16 lchunk[4U];
                    for (uint32_t _i = 0U; _i < 4U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t lane = 0U;
                    for (; lane < 4U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf02 * 4U * 4U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 4U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 +
                            __bfloat162float(kx) * __bfloat162float(__anf11);
                    }
                }
            }
        }
    }
    uint32_t i_kpr73 = 0U;
    for (; i_kpr73 < 2U; i_kpr73++) {
        uint32_t j_kpr74 = i_kpr73;
        uint32_t off = n_idx + j_kpr74 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr74 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_bf16_bf16_f32_f32_32x32x4_on
*/
static void
__hoisted_g_spmm_bf16_bf16_f32_f32_32x32x4_on_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix____nv_bfloat16 gA,
    uint32_t cols, __nv_bfloat16 *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    __nv_bfloat16 *elems_tile0 = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(__nv_bfloat16) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr75 = 0U;
        for (; i_kpr75 < 1U; i_kpr75++) {
            uint32_t j_kpr76 = i_kpr75;
            vec_memcpy(elems_tile0 + (j_kpr76 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr76 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr77 = 0U;
        for (; i_kpr77 < 2U; i_kpr77++) {
            uint32_t j_kpr78 = i_kpr77;
            vec_memcpy(col_ind_tile0 + (j_kpr78 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr78 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr79 = 0U;
        for (; i_kpr79 < to_; i_kpr79++) {
            uint32_t j_kpr80 = i_kpr79;
            elems_tile0[j_kpr80 * 4U + threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                __nv_bfloat16 kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        __nv_bfloat16 lchunk[4U];
                        for (uint32_t _i = 0U; _i < 4U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t lane = 0U;
                        for (; lane < 4U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf01 * 4U * 4U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 4U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 + __bfloat162float(kx) *
                                             __bfloat162float(__anf11);
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
            uint32_t i_kpr81 = 0U;
            for (; i_kpr81 < 1U; i_kpr81++) {
                uint32_t j_kpr82 = i_kpr81;
                vec_memcpy(elems_tile0 + (j_kpr82 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr82 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr83 = 0U;
            for (; i_kpr83 < 2U; i_kpr83++) {
                uint32_t j_kpr84 = i_kpr83;
                vec_memcpy(col_ind_tile0 + (j_kpr84 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr84 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    __nv_bfloat16 kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            __nv_bfloat16 lchunk[4U];
                            for (uint32_t _i = 0U; _i < 4U; ++_i)
                                lchunk[_i] = __float2bfloat16(0.0f);
                            uint32_t lane = 0U;
                            for (; lane < 4U; lane++)
                                lchunk[lane] =
                                    gB[kr * cols +
                                        (n_idx + __anf03 * 4U * 4U + lane)];
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                float __anf3 = out[__anf1 * 4U + ix];
                                __nv_bfloat16 __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 + __bfloat162float(kx) *
                                                 __bfloat162float(__anf11);
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
    uint32_t hi_kpr86 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr85 = 0U;
    for (; i_kpr85 < hi_kpr86; i_kpr85++) {
        uint32_t j_kpr87 = i_kpr85;
        elems_tile0[j_kpr87 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr87 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr87 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr87 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            __nv_bfloat16 kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    __nv_bfloat16 lchunk[4U];
                    for (uint32_t _i = 0U; _i < 4U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t lane = 0U;
                    for (; lane < 4U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf02 * 4U * 4U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 4U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 +
                            __bfloat162float(kx) * __bfloat162float(__anf11);
                    }
                }
            }
        }
    }
    uint32_t i_kpr88 = 0U;
    for (; i_kpr88 < 2U; i_kpr88++) {
        uint32_t j_kpr89 = i_kpr88;
        uint32_t off = n_idx + j_kpr89 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr89 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_f32_f16_f32_32x32x4
*/
static void
__hoisted_g_spmm_f16_f32_f16_f32_32x32x4_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols, float *gB,
    half *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr90 = 0U;
        for (; i_kpr90 < 1U; i_kpr90++) {
            uint32_t j_kpr91 = i_kpr90;
            vec_memcpy(elems_tile0 + (j_kpr91 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr91 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr92 = 0U;
        for (; i_kpr92 < 2U; i_kpr92++) {
            uint32_t j_kpr93 = i_kpr92;
            vec_memcpy(col_ind_tile0 + (j_kpr93 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr93 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr94 = 0U;
        for (; i_kpr94 < to_; i_kpr94++) {
            uint32_t j_kpr95 = i_kpr94;
            elems_tile0[j_kpr95 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[8U];
                    memset(lchunk, 0U, 8U * sizeof(float));
                    uint32_t lane = 0U;
                    for (; lane < 8U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf01 * 4U * 8U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        float __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + __half2float(kx) * __anf11;
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
                vec_memcpy(elems_tile0 + (j_kpr97 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr97 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr98 = 0U;
            for (; i_kpr98 < 2U; i_kpr98++) {
                uint32_t j_kpr99 = i_kpr98;
                vec_memcpy(col_ind_tile0 + (j_kpr99 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr99 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[8U];
                        memset(lchunk, 0U, 8U * sizeof(float));
                        uint32_t lane = 0U;
                        for (; lane < 8U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf03 * 4U * 8U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            float __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + __half2float(kx) * __anf11;
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
    uint32_t hi_kpr101 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr100 = 0U;
    for (; i_kpr100 < hi_kpr101; i_kpr100++) {
        uint32_t j_kpr102 = i_kpr100;
        elems_tile0[j_kpr102 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr102 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr102 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr102 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[8U];
                memset(lchunk, 0U, 8U * sizeof(float));
                uint32_t lane = 0U;
                for (; lane < 8U; lane++)
                    lchunk[lane] =
                        gB[kr * cols + (n_idx + __anf02 * 4U * 8U + lane)];
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    float __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + __half2float(kx) * __anf11;
                }
            }
        }
    }
    uint32_t i_kpr103 = 0U;
    for (; i_kpr103 < 1U; i_kpr103++) {
        uint32_t j_kpr104 = i_kpr103;
        uint32_t off = n_idx + j_kpr104 * 4U * 8U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            half combined[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                combined[_i] = __float2half_rn(0.0f);
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 8U; lane++) {
                uint32_t __anf4 = lane;
                combined[__anf4] = __float2half_rn(out[j_kpr104 * 8U + lane]);
            }
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_f32_f16_f32_32x32x4_on
*/
static void
__hoisted_g_spmm_f16_f32_f16_f32_32x32x4_on_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    float *gB, half *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr105 = 0U;
        for (; i_kpr105 < 1U; i_kpr105++) {
            uint32_t j_kpr106 = i_kpr105;
            vec_memcpy(elems_tile0 + (j_kpr106 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr106 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr107 = 0U;
        for (; i_kpr107 < 2U; i_kpr107++) {
            uint32_t j_kpr108 = i_kpr107;
            vec_memcpy(col_ind_tile0 + (j_kpr108 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr108 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr109 = 0U;
        for (; i_kpr109 < to_; i_kpr109++) {
            uint32_t j_kpr110 = i_kpr109;
            elems_tile0[j_kpr110 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[8U];
                    memset(lchunk, 0U, 8U * sizeof(float));
                    uint32_t lane = 0U;
                    for (; lane < 8U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf01 * 4U * 8U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        float __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + __half2float(kx) * __anf11;
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
            for (; i_kpr111 < 1U; i_kpr111++) {
                uint32_t j_kpr112 = i_kpr111;
                vec_memcpy(elems_tile0 + (j_kpr112 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr112 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr113 = 0U;
            for (; i_kpr113 < 2U; i_kpr113++) {
                uint32_t j_kpr114 = i_kpr113;
                vec_memcpy(col_ind_tile0 + (j_kpr114 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr114 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[8U];
                        memset(lchunk, 0U, 8U * sizeof(float));
                        uint32_t lane = 0U;
                        for (; lane < 8U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf03 * 4U * 8U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            float __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + __half2float(kx) * __anf11;
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
    uint32_t hi_kpr116 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr115 = 0U;
    for (; i_kpr115 < hi_kpr116; i_kpr115++) {
        uint32_t j_kpr117 = i_kpr115;
        elems_tile0[j_kpr117 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr117 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr117 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr117 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[8U];
                memset(lchunk, 0U, 8U * sizeof(float));
                uint32_t lane = 0U;
                for (; lane < 8U; lane++)
                    lchunk[lane] =
                        gB[kr * cols + (n_idx + __anf02 * 4U * 8U + lane)];
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    float __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + __half2float(kx) * __anf11;
                }
            }
        }
    }
    uint32_t i_kpr118 = 0U;
    for (; i_kpr118 < 1U; i_kpr118++) {
        uint32_t j_kpr119 = i_kpr118;
        uint32_t off = n_idx + j_kpr119 * 4U * 8U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            half combined[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                combined[_i] = __float2half_rn(0.0f);
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 8U; lane++) {
                uint32_t __anf4 = lane;
                combined[__anf4] = __float2half_rn(out[j_kpr119 * 8U + lane]);
            }
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_bf16_f32_f64_32x32x4
*/
static void
__hoisted_g_spmm_f16_bf16_f32_f64_32x32x4_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    __nv_bfloat16 *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    double out[8U];
    memset(out, 0U, 8U * sizeof(double));
    if (nnz >= 32U) {
        uint32_t i_kpr120 = 0U;
        for (; i_kpr120 < 1U; i_kpr120++) {
            uint32_t j_kpr121 = i_kpr120;
            vec_memcpy(elems_tile0 + (j_kpr121 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr121 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr122 = 0U;
        for (; i_kpr122 < 2U; i_kpr122++) {
            uint32_t j_kpr123 = i_kpr122;
            vec_memcpy(col_ind_tile0 + (j_kpr123 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr123 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr124 = 0U;
        for (; i_kpr124 < to_; i_kpr124++) {
            uint32_t j_kpr125 = i_kpr124;
            elems_tile0[j_kpr125 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        __nv_bfloat16 lchunk[4U];
                        for (uint32_t _i = 0U; _i < 4U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t lane = 0U;
                        for (; lane < 4U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf01 * 4U * 4U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            double __anf3 = out[__anf1 * 4U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 + (double) __half2float(kx) *
                                             (double) __bfloat162float(__anf11);
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
            uint32_t i_kpr126 = 0U;
            for (; i_kpr126 < 1U; i_kpr126++) {
                uint32_t j_kpr127 = i_kpr126;
                vec_memcpy(elems_tile0 + (j_kpr127 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr127 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr128 = 0U;
            for (; i_kpr128 < 2U; i_kpr128++) {
                uint32_t j_kpr129 = i_kpr128;
                vec_memcpy(col_ind_tile0 + (j_kpr129 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr129 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            __nv_bfloat16 lchunk[4U];
                            for (uint32_t _i = 0U; _i < 4U; ++_i)
                                lchunk[_i] = __float2bfloat16(0.0f);
                            uint32_t lane = 0U;
                            for (; lane < 4U; lane++)
                                lchunk[lane] =
                                    gB[kr * cols +
                                        (n_idx + __anf03 * 4U * 4U + lane)];
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                double __anf3 = out[__anf1 * 4U + ix];
                                __nv_bfloat16 __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 +
                                    (double) __half2float(kx) *
                                        (double) __bfloat162float(__anf11);
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
    uint32_t hi_kpr131 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr130 = 0U;
    for (; i_kpr130 < hi_kpr131; i_kpr130++) {
        uint32_t j_kpr132 = i_kpr130;
        elems_tile0[j_kpr132 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr132 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr132 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr132 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    __nv_bfloat16 lchunk[4U];
                    for (uint32_t _i = 0U; _i < 4U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t lane = 0U;
                    for (; lane < 4U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf02 * 4U * 4U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        double __anf3 = out[__anf1 * 4U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 + (double) __half2float(kx) *
                                         (double) __bfloat162float(__anf11);
                    }
                }
            }
        }
    }
    uint32_t i_kpr133 = 0U;
    for (; i_kpr133 < 2U; i_kpr133++) {
        uint32_t j_kpr134 = i_kpr133;
        uint32_t off = n_idx + j_kpr134 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = (float) out[j_kpr134 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_bf16_f32_f64_32x32x4_on
*/
static void
__hoisted_g_spmm_f16_bf16_f32_f64_32x32x4_on_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    __nv_bfloat16 *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    double out[8U];
    memset(out, 0U, 8U * sizeof(double));
    if (nnz >= 32U) {
        uint32_t i_kpr135 = 0U;
        for (; i_kpr135 < 1U; i_kpr135++) {
            uint32_t j_kpr136 = i_kpr135;
            vec_memcpy(elems_tile0 + (j_kpr136 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr136 * 4U + threadIdx.x) * 8U));
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
        for (; i_kpr139 < to_; i_kpr139++) {
            uint32_t j_kpr140 = i_kpr139;
            elems_tile0[j_kpr140 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        __nv_bfloat16 lchunk[4U];
                        for (uint32_t _i = 0U; _i < 4U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t lane = 0U;
                        for (; lane < 4U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf01 * 4U * 4U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            double __anf3 = out[__anf1 * 4U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 + (double) __half2float(kx) *
                                             (double) __bfloat162float(__anf11);
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
            uint32_t i_kpr141 = 0U;
            for (; i_kpr141 < 1U; i_kpr141++) {
                uint32_t j_kpr142 = i_kpr141;
                vec_memcpy(elems_tile0 + (j_kpr142 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr142 * 4U + threadIdx.x) * 8U));
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
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            __nv_bfloat16 lchunk[4U];
                            for (uint32_t _i = 0U; _i < 4U; ++_i)
                                lchunk[_i] = __float2bfloat16(0.0f);
                            uint32_t lane = 0U;
                            for (; lane < 4U; lane++)
                                lchunk[lane] =
                                    gB[kr * cols +
                                        (n_idx + __anf03 * 4U * 4U + lane)];
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                double __anf3 = out[__anf1 * 4U + ix];
                                __nv_bfloat16 __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 +
                                    (double) __half2float(kx) *
                                        (double) __bfloat162float(__anf11);
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
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    __nv_bfloat16 lchunk[4U];
                    for (uint32_t _i = 0U; _i < 4U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t lane = 0U;
                    for (; lane < 4U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf02 * 4U * 4U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        double __anf3 = out[__anf1 * 4U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 + (double) __half2float(kx) *
                                         (double) __bfloat162float(__anf11);
                    }
                }
            }
        }
    }
    uint32_t i_kpr148 = 0U;
    for (; i_kpr148 < 2U; i_kpr148++) {
        uint32_t j_kpr149 = i_kpr148;
        uint32_t off = n_idx + j_kpr149 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = (float) out[j_kpr149 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_f32_f32_f64_32x32x4
*/
static void
__hoisted_g_spmm_f16_f32_f32_f64_32x32x4_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols, float *gB,
    float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    double out[8U];
    memset(out, 0U, 8U * sizeof(double));
    if (nnz >= 32U) {
        uint32_t i_kpr150 = 0U;
        for (; i_kpr150 < 1U; i_kpr150++) {
            uint32_t j_kpr151 = i_kpr150;
            vec_memcpy(elems_tile0 + (j_kpr151 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr151 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr152 = 0U;
        for (; i_kpr152 < 2U; i_kpr152++) {
            uint32_t j_kpr153 = i_kpr152;
            vec_memcpy(col_ind_tile0 + (j_kpr153 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr153 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr154 = 0U;
        for (; i_kpr154 < to_; i_kpr154++) {
            uint32_t j_kpr155 = i_kpr154;
            elems_tile0[j_kpr155 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            double __anf3 = out[__anf1 * 4U + ix];
                            float __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 +
                                (double) __half2float(kx) * (double) __anf11;
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
            uint32_t i_kpr156 = 0U;
            for (; i_kpr156 < 1U; i_kpr156++) {
                uint32_t j_kpr157 = i_kpr156;
                vec_memcpy(elems_tile0 + (j_kpr157 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr157 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr158 = 0U;
            for (; i_kpr158 < 2U; i_kpr158++) {
                uint32_t j_kpr159 = i_kpr158;
                vec_memcpy(col_ind_tile0 + (j_kpr159 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr159 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                double __anf3 = out[__anf1 * 4U + ix];
                                float __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 + (double) __half2float(kx) *
                                                 (double) __anf11;
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
    uint32_t hi_kpr161 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr160 = 0U;
    for (; i_kpr160 < hi_kpr161; i_kpr160++) {
        uint32_t j_kpr162 = i_kpr160;
        elems_tile0[j_kpr162 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr162 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr162 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr162 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + (n_idx + __anf02 * 4U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        double __anf3 = out[__anf1 * 4U + ix];
                        float __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 +
                            (double) __half2float(kx) * (double) __anf11;
                    }
                }
            }
        }
    }
    uint32_t i_kpr163 = 0U;
    for (; i_kpr163 < 2U; i_kpr163++) {
        uint32_t j_kpr164 = i_kpr163;
        uint32_t off = n_idx + j_kpr164 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = (float) out[j_kpr164 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f16_f32_f32_f64_32x32x4_on
*/
static void
__hoisted_g_spmm_f16_f32_f32_f64_32x32x4_on_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    float *gB, float *gC)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 4U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    double out[8U];
    memset(out, 0U, 8U * sizeof(double));
    if (nnz >= 32U) {
        uint32_t i_kpr165 = 0U;
        for (; i_kpr165 < 1U; i_kpr165++) {
            uint32_t j_kpr166 = i_kpr165;
            vec_memcpy(elems_tile0 + (j_kpr166 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr166 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr167 = 0U;
        for (; i_kpr167 < 2U; i_kpr167++) {
            uint32_t j_kpr168 = i_kpr167;
            vec_memcpy(col_ind_tile0 + (j_kpr168 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr168 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr169 = 0U;
        for (; i_kpr169 < to_; i_kpr169++) {
            uint32_t j_kpr170 = i_kpr169;
            elems_tile0[j_kpr170 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 2U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    if (n_idx + __anf01 * 4U * 4U < cols) {
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++) {
                            uint32_t __anf4 = ix;
                            double __anf3 = out[__anf1 * 4U + ix];
                            float __anf11 = lchunk[ix];
                            out[__anf1 * 4U + __anf4] =
                                __anf3 +
                                (double) __half2float(kx) * (double) __anf11;
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
                vec_memcpy(elems_tile0 + (j_kpr172 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr172 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr173 = 0U;
            for (; i_kpr173 < 2U; i_kpr173++) {
                uint32_t j_kpr174 = i_kpr173;
                vec_memcpy(col_ind_tile0 + (j_kpr174 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr174 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 2U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        if (n_idx + __anf03 * 4U * 4U < cols) {
                            float lchunk[4U];
                            memset(lchunk, 0U, 4U * sizeof(float));
                            vec_memcpy(lchunk,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++) {
                                uint32_t __anf4 = ix;
                                double __anf3 = out[__anf1 * 4U + ix];
                                float __anf11 = lchunk[ix];
                                out[__anf1 * 4U + __anf4] =
                                    __anf3 + (double) __half2float(kx) *
                                                 (double) __anf11;
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
    uint32_t hi_kpr176 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr175 = 0U;
    for (; i_kpr175 < hi_kpr176; i_kpr175++) {
        uint32_t j_kpr177 = i_kpr175;
        elems_tile0[j_kpr177 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr177 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr177 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr177 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 4U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(
                        lchunk, gB + (cols * kr + (n_idx + __anf02 * 4U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++) {
                        uint32_t __anf4 = ix;
                        double __anf3 = out[__anf1 * 4U + ix];
                        float __anf11 = lchunk[ix];
                        out[__anf1 * 4U + __anf4] =
                            __anf3 +
                            (double) __half2float(kx) * (double) __anf11;
                    }
                }
            }
        }
    }
    uint32_t i_kpr178 = 0U;
    for (; i_kpr178 < 2U; i_kpr178++) {
        uint32_t j_kpr179 = i_kpr178;
        uint32_t off = n_idx + j_kpr179 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = (float) out[j_kpr179 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_mapped_f16_f32_f16_f32_32x32x4
*/
static void
__hoisted_g_spmm_mapped_f16_f32_f16_f32_32x32x4_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    float *gB, half *gC, float beta, float alpha)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr180 = 0U;
        for (; i_kpr180 < 1U; i_kpr180++) {
            uint32_t j_kpr181 = i_kpr180;
            vec_memcpy(elems_tile0 + (j_kpr181 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr181 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr182 = 0U;
        for (; i_kpr182 < 2U; i_kpr182++) {
            uint32_t j_kpr183 = i_kpr182;
            vec_memcpy(col_ind_tile0 + (j_kpr183 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr183 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr184 = 0U;
        for (; i_kpr184 < to_; i_kpr184++) {
            uint32_t j_kpr185 = i_kpr184;
            elems_tile0[j_kpr185 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[8U];
                    memset(lchunk, 0U, 8U * sizeof(float));
                    uint32_t lane = 0U;
                    for (; lane < 8U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf01 * 4U * 8U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        float __anf11 = lchunk[ix];
                        float y = __half2float(kx);
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + (y + y) * (__anf11 + 1.0f);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr186 = 0U;
            for (; i_kpr186 < 1U; i_kpr186++) {
                uint32_t j_kpr187 = i_kpr186;
                vec_memcpy(elems_tile0 + (j_kpr187 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr187 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr188 = 0U;
            for (; i_kpr188 < 2U; i_kpr188++) {
                uint32_t j_kpr189 = i_kpr188;
                vec_memcpy(col_ind_tile0 + (j_kpr189 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr189 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[8U];
                        memset(lchunk, 0U, 8U * sizeof(float));
                        uint32_t lane = 0U;
                        for (; lane < 8U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf03 * 4U * 8U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            float __anf11 = lchunk[ix];
                            float y = __half2float(kx);
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + (y + y) * (__anf11 + 1.0f);
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
    uint32_t hi_kpr191 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr190 = 0U;
    for (; i_kpr190 < hi_kpr191; i_kpr190++) {
        uint32_t j_kpr192 = i_kpr190;
        elems_tile0[j_kpr192 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr192 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr192 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr192 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[8U];
                memset(lchunk, 0U, 8U * sizeof(float));
                uint32_t lane = 0U;
                for (; lane < 8U; lane++)
                    lchunk[lane] =
                        gB[kr * cols + (n_idx + __anf02 * 4U * 8U + lane)];
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    float __anf11 = lchunk[ix];
                    float y = __half2float(kx);
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + (y + y) * (__anf11 + 1.0f);
                }
            }
        }
    }
    uint32_t i_kpr193 = 0U;
    for (; i_kpr193 < 1U; i_kpr193++) {
        uint32_t j_kpr194 = i_kpr193;
        uint32_t off = n_idx + j_kpr194 * 4U * 8U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            half combined[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                combined[_i] = __float2half_rn(0.0f);
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 8U; lane++) {
                uint32_t __anf4 = lane;
                float __anf1 = out[j_kpr194 * 8U + lane];
                combined[__anf4] = __float2half_rn(
                    beta * __half2float(combined[lane]) + alpha * __anf1);
            }
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_mapped_f16_f32_f16_f32_32x32x4_on
*/
static void
__hoisted_g_spmm_mapped_f16_f32_f16_f32_32x32x4_on_0(uint32_t *row_indices,
    uint32_t rows, Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t cols,
    float *gB, half *gC, float beta, float alpha)
{
    uint32_t m_idx = row_indices[blockIdx.x % rows];
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
    half *elems_tile0 = (half *) KPR_SHMEM_AT(0U);
    uint32_t *col_ind_tile0 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(half) * 32U);
    uint32_t ri = gA.row_off[m_idx];
    uint32_t re = gA.row_off[m_idx + 1U];
    uint32_t ri_ = ri / 8U * 8U;
    uint32_t nnz = re - ri_;
    uint32_t idx = 0U;
    float out[8U];
    memset(out, 0U, 8U * sizeof(float));
    if (nnz >= 32U) {
        uint32_t i_kpr195 = 0U;
        for (; i_kpr195 < 1U; i_kpr195++) {
            uint32_t j_kpr196 = i_kpr195;
            vec_memcpy(elems_tile0 + (j_kpr196 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr196 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr197 = 0U;
        for (; i_kpr197 < 2U; i_kpr197++) {
            uint32_t j_kpr198 = i_kpr197;
            vec_memcpy(col_ind_tile0 + (j_kpr198 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr198 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr199 = 0U;
        for (; i_kpr199 < to_; i_kpr199++) {
            uint32_t j_kpr200 = i_kpr199;
            elems_tile0[j_kpr200 * 4U + threadIdx.x] = __float2half_rn(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                half kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[8U];
                    memset(lchunk, 0U, 8U * sizeof(float));
                    uint32_t lane = 0U;
                    for (; lane < 8U; lane++)
                        lchunk[lane] =
                            gB[kr * cols + (n_idx + __anf01 * 4U * 8U + lane)];
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        float __anf11 = lchunk[ix];
                        float y = __half2float(kx);
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + (y + y) * (__anf11 + 1.0f);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr201 = 0U;
            for (; i_kpr201 < 1U; i_kpr201++) {
                uint32_t j_kpr202 = i_kpr201;
                vec_memcpy(elems_tile0 + (j_kpr202 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr202 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr203 = 0U;
            for (; i_kpr203 < 2U; i_kpr203++) {
                uint32_t j_kpr204 = i_kpr203;
                vec_memcpy(col_ind_tile0 + (j_kpr204 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr204 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    half kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[8U];
                        memset(lchunk, 0U, 8U * sizeof(float));
                        uint32_t lane = 0U;
                        for (; lane < 8U; lane++)
                            lchunk[lane] =
                                gB[kr * cols +
                                    (n_idx + __anf03 * 4U * 8U + lane)];
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            float __anf11 = lchunk[ix];
                            float y = __half2float(kx);
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + (y + y) * (__anf11 + 1.0f);
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
    uint32_t hi_kpr206 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr205 = 0U;
    for (; i_kpr205 < hi_kpr206; i_kpr205++) {
        uint32_t j_kpr207 = i_kpr205;
        elems_tile0[j_kpr207 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr207 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr207 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr207 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            half kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[8U];
                memset(lchunk, 0U, 8U * sizeof(float));
                uint32_t lane = 0U;
                for (; lane < 8U; lane++)
                    lchunk[lane] =
                        gB[kr * cols + (n_idx + __anf02 * 4U * 8U + lane)];
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    float __anf11 = lchunk[ix];
                    float y = __half2float(kx);
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + (y + y) * (__anf11 + 1.0f);
                }
            }
        }
    }
    uint32_t i_kpr208 = 0U;
    for (; i_kpr208 < 1U; i_kpr208++) {
        uint32_t j_kpr209 = i_kpr208;
        uint32_t off = n_idx + j_kpr209 * 4U * 8U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            half combined[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                combined[_i] = __float2half_rn(0.0f);
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 8U; lane++) {
                uint32_t __anf4 = lane;
                float __anf1 = out[j_kpr209 * 8U + lane];
                combined[__anf4] = __float2half_rn(
                    beta * __half2float(combined[lane]) + alpha * __anf1);
            }
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting g_spmm_f32_32x4x1
*/
static void
__hoisted_g_spmm_f32_32x4x1_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr210 = 0U;
        for (; i_kpr210 < 8U; i_kpr210++) {
            uint32_t j_kpr211 = i_kpr210;
            vec_memcpy(elems_tile0 + (j_kpr211 + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr211 + threadIdx.x) * 4U));
        }
        uint32_t i_kpr212 = 0U;
        for (; i_kpr212 < 8U; i_kpr212++) {
            uint32_t j_kpr213 = i_kpr212;
            vec_memcpy(col_ind_tile0 + (j_kpr213 + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr213 + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i_kpr214 = 0U;
        for (; i_kpr214 < to_; i_kpr214++)
            elems_tile0[i_kpr214 + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr216 = 0U;
            for (; i_kpr216 < 8U; i_kpr216++) {
                uint32_t j_kpr217 = i_kpr216;
                vec_memcpy(elems_tile0 + (j_kpr217 + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr217 + threadIdx.x) * 4U));
            }
            uint32_t i_kpr218 = 0U;
            for (; i_kpr218 < 8U; i_kpr218++) {
                uint32_t j_kpr219 = i_kpr218;
                vec_memcpy(col_ind_tile0 + (j_kpr219 + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr219 + threadIdx.x) * 4U));
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
                            lchunk, gB + (cols * kr + (n_idx + __anf03 * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr221 = __anf01 - threadIdx.x;
    uint32_t i_kpr220 = 0U;
    for (; i_kpr220 < hi_kpr221; i_kpr220++) {
        uint32_t j_kpr222 = i_kpr220;
        elems_tile0[j_kpr222 + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr222 + threadIdx.x];
        col_ind_tile0[j_kpr222 + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr222 + threadIdx.x];
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
                vec_memcpy(lchunk, gB + (cols * kr + (n_idx + __anf02 * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr223 = 0U;
    for (; i_kpr223 < 1U; i_kpr223++) {
        uint32_t j_kpr224 = i_kpr223;
        uint32_t off = n_idx + j_kpr224 * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr224 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(2)
/**
  hoisted when extracting g_spmm_f32_32x8x2
*/
static void
__hoisted_g_spmm_f32_32x8x2_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr225 = 0U;
        for (; i_kpr225 < 4U; i_kpr225++) {
            uint32_t j_kpr226 = i_kpr225;
            vec_memcpy(elems_tile0 + (j_kpr226 * 2U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr226 * 2U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr227 = 0U;
        for (; i_kpr227 < 4U; i_kpr227++) {
            uint32_t j_kpr228 = i_kpr227;
            vec_memcpy(col_ind_tile0 + (j_kpr228 * 2U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr228 * 2U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i_kpr229 = 0U;
        for (; i_kpr229 < to_; i_kpr229++)
            elems_tile0[i_kpr229 * 2U + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 2U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr231 = 0U;
            for (; i_kpr231 < 4U; i_kpr231++) {
                uint32_t j_kpr232 = i_kpr231;
                vec_memcpy(elems_tile0 + (j_kpr232 * 2U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr232 * 2U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr233 = 0U;
            for (; i_kpr233 < 4U; i_kpr233++) {
                uint32_t j_kpr234 = i_kpr233;
                vec_memcpy(col_ind_tile0 + (j_kpr234 * 2U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr234 * 2U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 2U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr236 = (__anf01 + 1U - threadIdx.x) / 2U;
    uint32_t i_kpr235 = 0U;
    for (; i_kpr235 < hi_kpr236; i_kpr235++) {
        uint32_t j_kpr237 = i_kpr235;
        elems_tile0[j_kpr237 * 2U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr237 * 2U + threadIdx.x];
        col_ind_tile0[j_kpr237 * 2U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr237 * 2U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 2U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr238 = 0U;
    for (; i_kpr238 < 1U; i_kpr238++) {
        uint32_t j_kpr239 = i_kpr238;
        uint32_t off = n_idx + j_kpr239 * 2U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr239 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f32_32x16x4
*/
static void
__hoisted_g_spmm_f32_32x16x4_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr240 = 0U;
        for (; i_kpr240 < 2U; i_kpr240++) {
            uint32_t j_kpr241 = i_kpr240;
            vec_memcpy(elems_tile0 + (j_kpr241 * 4U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr241 * 4U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr242 = 0U;
        for (; i_kpr242 < 2U; i_kpr242++) {
            uint32_t j_kpr243 = i_kpr242;
            vec_memcpy(col_ind_tile0 + (j_kpr243 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr243 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr244 = 0U;
        for (; i_kpr244 < to_; i_kpr244++)
            elems_tile0[i_kpr244 * 4U + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 4U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr246 = 0U;
            for (; i_kpr246 < 2U; i_kpr246++) {
                uint32_t j_kpr247 = i_kpr246;
                vec_memcpy(elems_tile0 + (j_kpr247 * 4U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr247 * 4U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr248 = 0U;
            for (; i_kpr248 < 2U; i_kpr248++) {
                uint32_t j_kpr249 = i_kpr248;
                vec_memcpy(col_ind_tile0 + (j_kpr249 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr249 * 4U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 4U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr251 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr250 = 0U;
    for (; i_kpr250 < hi_kpr251; i_kpr250++) {
        uint32_t j_kpr252 = i_kpr250;
        elems_tile0[j_kpr252 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr252 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr252 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr252 * 4U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 4U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr253 = 0U;
    for (; i_kpr253 < 1U; i_kpr253++) {
        uint32_t j_kpr254 = i_kpr253;
        uint32_t off = n_idx + j_kpr254 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr254 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x32x8
*/
static void
__hoisted_g_spmm_f32_32x32x8_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr255 = 0U;
        for (; i_kpr255 < 1U; i_kpr255++) {
            uint32_t j_kpr256 = i_kpr255;
            vec_memcpy(elems_tile0 + (j_kpr256 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr256 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr257 = 0U;
        for (; i_kpr257 < 1U; i_kpr257++) {
            uint32_t j_kpr258 = i_kpr257;
            vec_memcpy(col_ind_tile0 + (j_kpr258 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr258 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr259 = 0U;
        for (; i_kpr259 < to_; i_kpr259++)
            elems_tile0[i_kpr259 * 8U + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 8U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr261 = 0U;
            for (; i_kpr261 < 1U; i_kpr261++) {
                uint32_t j_kpr262 = i_kpr261;
                vec_memcpy(elems_tile0 + (j_kpr262 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr262 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr263 = 0U;
            for (; i_kpr263 < 1U; i_kpr263++) {
                uint32_t j_kpr264 = i_kpr263;
                vec_memcpy(col_ind_tile0 + (j_kpr264 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr264 * 8U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 8U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr266 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr265 = 0U;
    for (; i_kpr265 < hi_kpr266; i_kpr265++) {
        uint32_t j_kpr267 = i_kpr265;
        elems_tile0[j_kpr267 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr267 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr267 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr267 * 8U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 8U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr268 = 0U;
    for (; i_kpr268 < 1U; i_kpr268++) {
        uint32_t j_kpr269 = i_kpr268;
        uint32_t off = n_idx + j_kpr269 * 8U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr269 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x64x8
*/
static void
__hoisted_g_spmm_f32_32x64x8_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr270 = 0U;
        for (; i_kpr270 < 1U; i_kpr270++) {
            uint32_t j_kpr271 = i_kpr270;
            vec_memcpy(elems_tile0 + (j_kpr271 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr271 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr272 = 0U;
        for (; i_kpr272 < 1U; i_kpr272++) {
            uint32_t j_kpr273 = i_kpr272;
            vec_memcpy(col_ind_tile0 + (j_kpr273 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr273 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr274 = 0U;
        for (; i_kpr274 < to_; i_kpr274++)
            elems_tile0[i_kpr274 * 8U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 8U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr276 = 0U;
            for (; i_kpr276 < 1U; i_kpr276++) {
                uint32_t j_kpr277 = i_kpr276;
                vec_memcpy(elems_tile0 + (j_kpr277 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr277 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr278 = 0U;
            for (; i_kpr278 < 1U; i_kpr278++) {
                uint32_t j_kpr279 = i_kpr278;
                vec_memcpy(col_ind_tile0 + (j_kpr279 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr279 * 8U + threadIdx.x) * 4U));
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
                                gB + (cols * kr + (n_idx + __anf03 * 8U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr281 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr280 = 0U;
    for (; i_kpr280 < hi_kpr281; i_kpr280++) {
        uint32_t j_kpr282 = i_kpr280;
        elems_tile0[j_kpr282 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr282 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr282 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr282 * 8U + threadIdx.x];
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
                        lchunk, gB + (cols * kr + (n_idx + __anf02 * 8U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr283 = 0U;
    for (; i_kpr283 < 2U; i_kpr283++) {
        uint32_t j_kpr284 = i_kpr283;
        uint32_t off = n_idx + j_kpr284 * 8U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr284 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting g_spmm_f32_32x4x1_on
*/
static void
__hoisted_g_spmm_f32_32x4x1_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr285 = 0U;
        for (; i_kpr285 < 8U; i_kpr285++) {
            uint32_t j_kpr286 = i_kpr285;
            vec_memcpy(elems_tile0 + (j_kpr286 + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr286 + threadIdx.x) * 4U));
        }
        uint32_t i_kpr287 = 0U;
        for (; i_kpr287 < 8U; i_kpr287++) {
            uint32_t j_kpr288 = i_kpr287;
            vec_memcpy(col_ind_tile0 + (j_kpr288 + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr288 + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i_kpr289 = 0U;
        for (; i_kpr289 < to_; i_kpr289++)
            elems_tile0[i_kpr289 + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr291 = 0U;
            for (; i_kpr291 < 8U; i_kpr291++) {
                uint32_t j_kpr292 = i_kpr291;
                vec_memcpy(elems_tile0 + (j_kpr292 + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr292 + threadIdx.x) * 4U));
            }
            uint32_t i_kpr293 = 0U;
            for (; i_kpr293 < 8U; i_kpr293++) {
                uint32_t j_kpr294 = i_kpr293;
                vec_memcpy(col_ind_tile0 + (j_kpr294 + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr294 + threadIdx.x) * 4U));
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
                            lchunk, gB + (cols * kr + (n_idx + __anf03 * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr296 = __anf01 - threadIdx.x;
    uint32_t i_kpr295 = 0U;
    for (; i_kpr295 < hi_kpr296; i_kpr295++) {
        uint32_t j_kpr297 = i_kpr295;
        elems_tile0[j_kpr297 + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr297 + threadIdx.x];
        col_ind_tile0[j_kpr297 + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr297 + threadIdx.x];
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
                vec_memcpy(lchunk, gB + (cols * kr + (n_idx + __anf02 * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr298 = 0U;
    for (; i_kpr298 < 1U; i_kpr298++) {
        uint32_t j_kpr299 = i_kpr298;
        uint32_t off = n_idx + j_kpr299 * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr299 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(2)
/**
  hoisted when extracting g_spmm_f32_32x8x2_on
*/
static void
__hoisted_g_spmm_f32_32x8x2_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr300 = 0U;
        for (; i_kpr300 < 4U; i_kpr300++) {
            uint32_t j_kpr301 = i_kpr300;
            vec_memcpy(elems_tile0 + (j_kpr301 * 2U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr301 * 2U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr302 = 0U;
        for (; i_kpr302 < 4U; i_kpr302++) {
            uint32_t j_kpr303 = i_kpr302;
            vec_memcpy(col_ind_tile0 + (j_kpr303 * 2U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr303 * 2U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i_kpr304 = 0U;
        for (; i_kpr304 < to_; i_kpr304++)
            elems_tile0[i_kpr304 * 2U + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 2U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr306 = 0U;
            for (; i_kpr306 < 4U; i_kpr306++) {
                uint32_t j_kpr307 = i_kpr306;
                vec_memcpy(elems_tile0 + (j_kpr307 * 2U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr307 * 2U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr308 = 0U;
            for (; i_kpr308 < 4U; i_kpr308++) {
                uint32_t j_kpr309 = i_kpr308;
                vec_memcpy(col_ind_tile0 + (j_kpr309 * 2U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr309 * 2U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 2U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr311 = (__anf01 + 1U - threadIdx.x) / 2U;
    uint32_t i_kpr310 = 0U;
    for (; i_kpr310 < hi_kpr311; i_kpr310++) {
        uint32_t j_kpr312 = i_kpr310;
        elems_tile0[j_kpr312 * 2U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr312 * 2U + threadIdx.x];
        col_ind_tile0[j_kpr312 * 2U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr312 * 2U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 2U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr313 = 0U;
    for (; i_kpr313 < 1U; i_kpr313++) {
        uint32_t j_kpr314 = i_kpr313;
        uint32_t off = n_idx + j_kpr314 * 2U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr314 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_spmm_f32_32x16x4_on
*/
static void
__hoisted_g_spmm_f32_32x16x4_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr315 = 0U;
        for (; i_kpr315 < 2U; i_kpr315++) {
            uint32_t j_kpr316 = i_kpr315;
            vec_memcpy(elems_tile0 + (j_kpr316 * 4U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr316 * 4U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr317 = 0U;
        for (; i_kpr317 < 2U; i_kpr317++) {
            uint32_t j_kpr318 = i_kpr317;
            vec_memcpy(col_ind_tile0 + (j_kpr318 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr318 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr319 = 0U;
        for (; i_kpr319 < to_; i_kpr319++)
            elems_tile0[i_kpr319 * 4U + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 4U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr321 = 0U;
            for (; i_kpr321 < 2U; i_kpr321++) {
                uint32_t j_kpr322 = i_kpr321;
                vec_memcpy(elems_tile0 + (j_kpr322 * 4U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr322 * 4U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr323 = 0U;
            for (; i_kpr323 < 2U; i_kpr323++) {
                uint32_t j_kpr324 = i_kpr323;
                vec_memcpy(col_ind_tile0 + (j_kpr324 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr324 * 4U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 4U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr326 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr325 = 0U;
    for (; i_kpr325 < hi_kpr326; i_kpr325++) {
        uint32_t j_kpr327 = i_kpr325;
        elems_tile0[j_kpr327 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr327 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr327 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr327 * 4U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 4U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr328 = 0U;
    for (; i_kpr328 < 1U; i_kpr328++) {
        uint32_t j_kpr329 = i_kpr328;
        uint32_t off = n_idx + j_kpr329 * 4U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr329 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x32x8_on
*/
static void
__hoisted_g_spmm_f32_32x32x8_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr330 = 0U;
        for (; i_kpr330 < 1U; i_kpr330++) {
            uint32_t j_kpr331 = i_kpr330;
            vec_memcpy(elems_tile0 + (j_kpr331 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr331 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr332 = 0U;
        for (; i_kpr332 < 1U; i_kpr332++) {
            uint32_t j_kpr333 = i_kpr332;
            vec_memcpy(col_ind_tile0 + (j_kpr333 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr333 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr334 = 0U;
        for (; i_kpr334 < to_; i_kpr334++)
            elems_tile0[i_kpr334 * 8U + threadIdx.x] = 0.0f;
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
                        lchunk, gB + (cols * kr + (n_idx + __anf01 * 8U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr336 = 0U;
            for (; i_kpr336 < 1U; i_kpr336++) {
                uint32_t j_kpr337 = i_kpr336;
                vec_memcpy(elems_tile0 + (j_kpr337 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr337 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr338 = 0U;
            for (; i_kpr338 < 1U; i_kpr338++) {
                uint32_t j_kpr339 = i_kpr338;
                vec_memcpy(col_ind_tile0 + (j_kpr339 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr339 * 8U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 8U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr341 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr340 = 0U;
    for (; i_kpr340 < hi_kpr341; i_kpr340++) {
        uint32_t j_kpr342 = i_kpr340;
        elems_tile0[j_kpr342 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr342 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr342 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr342 * 8U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 8U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr343 = 0U;
    for (; i_kpr343 < 1U; i_kpr343++) {
        uint32_t j_kpr344 = i_kpr343;
        uint32_t off = n_idx + j_kpr344 * 8U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr344 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_spmm_f32_32x64x8_on
*/
static void
__hoisted_g_spmm_f32_32x64x8_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr345 = 0U;
        for (; i_kpr345 < 1U; i_kpr345++) {
            uint32_t j_kpr346 = i_kpr345;
            vec_memcpy(elems_tile0 + (j_kpr346 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr346 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr347 = 0U;
        for (; i_kpr347 < 1U; i_kpr347++) {
            uint32_t j_kpr348 = i_kpr347;
            vec_memcpy(col_ind_tile0 + (j_kpr348 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr348 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr349 = 0U;
        for (; i_kpr349 < to_; i_kpr349++)
            elems_tile0[i_kpr349 * 8U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 8U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr351 = 0U;
            for (; i_kpr351 < 1U; i_kpr351++) {
                uint32_t j_kpr352 = i_kpr351;
                vec_memcpy(elems_tile0 + (j_kpr352 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr352 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr353 = 0U;
            for (; i_kpr353 < 1U; i_kpr353++) {
                uint32_t j_kpr354 = i_kpr353;
                vec_memcpy(col_ind_tile0 + (j_kpr354 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr354 * 8U + threadIdx.x) * 4U));
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
                                gB + (cols * kr + (n_idx + __anf03 * 8U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr356 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr355 = 0U;
    for (; i_kpr355 < hi_kpr356; i_kpr355++) {
        uint32_t j_kpr357 = i_kpr355;
        elems_tile0[j_kpr357 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr357 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr357 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr357 * 8U + threadIdx.x];
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
                        lchunk, gB + (cols * kr + (n_idx + __anf02 * 8U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr358 = 0U;
    for (; i_kpr358 < 2U; i_kpr358++) {
        uint32_t j_kpr359 = i_kpr358;
        uint32_t off = n_idx + j_kpr359 * 8U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr359 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x64x16
*/
static void
__hoisted_g_spmm_f32_64x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr360 = 0U;
        for (; i_kpr360 < 1U; i_kpr360++) {
            uint32_t j_kpr361 = i_kpr360;
            vec_memcpy(elems_tile0 + (j_kpr361 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr361 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr362 = 0U;
        for (; i_kpr362 < 1U; i_kpr362++) {
            uint32_t j_kpr363 = i_kpr362;
            vec_memcpy(col_ind_tile0 + (j_kpr363 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr363 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr364 = 0U;
        for (; i_kpr364 < to_; i_kpr364++)
            elems_tile0[i_kpr364 * 16U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr366 = 0U;
            for (; i_kpr366 < 1U; i_kpr366++) {
                uint32_t j_kpr367 = i_kpr366;
                vec_memcpy(elems_tile0 + (j_kpr367 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr367 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr368 = 0U;
            for (; i_kpr368 < 1U; i_kpr368++) {
                uint32_t j_kpr369 = i_kpr368;
                vec_memcpy(col_ind_tile0 + (j_kpr369 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr369 * 16U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr371 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr370 = 0U;
    for (; i_kpr370 < hi_kpr371; i_kpr370++) {
        uint32_t j_kpr372 = i_kpr370;
        elems_tile0[j_kpr372 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr372 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr372 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr372 * 16U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr373 = 0U;
    for (; i_kpr373 < 1U; i_kpr373++) {
        uint32_t j_kpr374 = i_kpr373;
        uint32_t off = n_idx + j_kpr374 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr374 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x64x16_on
*/
static void
__hoisted_g_spmm_f32_64x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr375 = 0U;
        for (; i_kpr375 < 1U; i_kpr375++) {
            uint32_t j_kpr376 = i_kpr375;
            vec_memcpy(elems_tile0 + (j_kpr376 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr376 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr377 = 0U;
        for (; i_kpr377 < 1U; i_kpr377++) {
            uint32_t j_kpr378 = i_kpr377;
            vec_memcpy(col_ind_tile0 + (j_kpr378 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr378 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr379 = 0U;
        for (; i_kpr379 < to_; i_kpr379++)
            elems_tile0[i_kpr379 * 16U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr381 = 0U;
            for (; i_kpr381 < 1U; i_kpr381++) {
                uint32_t j_kpr382 = i_kpr381;
                vec_memcpy(elems_tile0 + (j_kpr382 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr382 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr383 = 0U;
            for (; i_kpr383 < 1U; i_kpr383++) {
                uint32_t j_kpr384 = i_kpr383;
                vec_memcpy(col_ind_tile0 + (j_kpr384 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr384 * 16U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr386 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr385 = 0U;
    for (; i_kpr385 < hi_kpr386; i_kpr385++) {
        uint32_t j_kpr387 = i_kpr385;
        elems_tile0[j_kpr387 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr387 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr387 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr387 * 16U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr388 = 0U;
    for (; i_kpr388 < 1U; i_kpr388++) {
        uint32_t j_kpr389 = i_kpr388;
        uint32_t off = n_idx + j_kpr389 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr389 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x128x16
*/
static void
__hoisted_g_spmm_f32_64x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr390 = 0U;
        for (; i_kpr390 < 1U; i_kpr390++) {
            uint32_t j_kpr391 = i_kpr390;
            vec_memcpy(elems_tile0 + (j_kpr391 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr391 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr392 = 0U;
        for (; i_kpr392 < 1U; i_kpr392++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr396 = 0U;
            for (; i_kpr396 < 1U; i_kpr396++) {
                uint32_t j_kpr397 = i_kpr396;
                vec_memcpy(elems_tile0 + (j_kpr397 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr397 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr398 = 0U;
            for (; i_kpr398 < 1U; i_kpr398++) {
                uint32_t j_kpr399 = i_kpr398;
                vec_memcpy(col_ind_tile0 + (j_kpr399 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr399 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr403 = 0U;
    for (; i_kpr403 < 2U; i_kpr403++) {
        uint32_t j_kpr404 = i_kpr403;
        uint32_t off = n_idx + j_kpr404 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr404 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x128x16_on
*/
static void
__hoisted_g_spmm_f32_64x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr405 = 0U;
        for (; i_kpr405 < 1U; i_kpr405++) {
            uint32_t j_kpr406 = i_kpr405;
            vec_memcpy(elems_tile0 + (j_kpr406 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr406 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr407 = 0U;
        for (; i_kpr407 < 1U; i_kpr407++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr411 = 0U;
            for (; i_kpr411 < 1U; i_kpr411++) {
                uint32_t j_kpr412 = i_kpr411;
                vec_memcpy(elems_tile0 + (j_kpr412 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr412 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr413 = 0U;
            for (; i_kpr413 < 1U; i_kpr413++) {
                uint32_t j_kpr414 = i_kpr413;
                vec_memcpy(col_ind_tile0 + (j_kpr414 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr414 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr418 = 0U;
    for (; i_kpr418 < 2U; i_kpr418++) {
        uint32_t j_kpr419 = i_kpr418;
        uint32_t off = n_idx + j_kpr419 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr419 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x256x16
*/
static void
__hoisted_g_spmm_f32_64x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr420 = 0U;
        for (; i_kpr420 < 1U; i_kpr420++) {
            uint32_t j_kpr421 = i_kpr420;
            vec_memcpy(elems_tile0 + (j_kpr421 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr421 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr422 = 0U;
        for (; i_kpr422 < 1U; i_kpr422++) {
            uint32_t j_kpr423 = i_kpr422;
            vec_memcpy(col_ind_tile0 + (j_kpr423 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr423 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr424 = 0U;
        for (; i_kpr424 < to_; i_kpr424++)
            elems_tile0[i_kpr424 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr426 = 0U;
            for (; i_kpr426 < 1U; i_kpr426++) {
                uint32_t j_kpr427 = i_kpr426;
                vec_memcpy(elems_tile0 + (j_kpr427 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr427 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr428 = 0U;
            for (; i_kpr428 < 1U; i_kpr428++) {
                uint32_t j_kpr429 = i_kpr428;
                vec_memcpy(col_ind_tile0 + (j_kpr429 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr429 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr431 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr430 = 0U;
    for (; i_kpr430 < hi_kpr431; i_kpr430++) {
        uint32_t j_kpr432 = i_kpr430;
        elems_tile0[j_kpr432 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr432 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr432 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr432 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr433 = 0U;
    for (; i_kpr433 < 4U; i_kpr433++) {
        uint32_t j_kpr434 = i_kpr433;
        uint32_t off = n_idx + j_kpr434 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr434 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x256x16_on
*/
static void
__hoisted_g_spmm_f32_64x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr435 = 0U;
        for (; i_kpr435 < 1U; i_kpr435++) {
            uint32_t j_kpr436 = i_kpr435;
            vec_memcpy(elems_tile0 + (j_kpr436 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr436 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr437 = 0U;
        for (; i_kpr437 < 1U; i_kpr437++) {
            uint32_t j_kpr438 = i_kpr437;
            vec_memcpy(col_ind_tile0 + (j_kpr438 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr438 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr439 = 0U;
        for (; i_kpr439 < to_; i_kpr439++)
            elems_tile0[i_kpr439 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr441 = 0U;
            for (; i_kpr441 < 1U; i_kpr441++) {
                uint32_t j_kpr442 = i_kpr441;
                vec_memcpy(elems_tile0 + (j_kpr442 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr442 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr443 = 0U;
            for (; i_kpr443 < 1U; i_kpr443++) {
                uint32_t j_kpr444 = i_kpr443;
                vec_memcpy(col_ind_tile0 + (j_kpr444 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr444 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr446 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr445 = 0U;
    for (; i_kpr445 < hi_kpr446; i_kpr445++) {
        uint32_t j_kpr447 = i_kpr445;
        elems_tile0[j_kpr447 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr447 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr447 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr447 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr448 = 0U;
    for (; i_kpr448 < 4U; i_kpr448++) {
        uint32_t j_kpr449 = i_kpr448;
        uint32_t off = n_idx + j_kpr449 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr449 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x512x16
*/
static void
__hoisted_g_spmm_f32_64x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr450 = 0U;
        for (; i_kpr450 < 1U; i_kpr450++) {
            uint32_t j_kpr451 = i_kpr450;
            vec_memcpy(elems_tile0 + (j_kpr451 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr451 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr452 = 0U;
        for (; i_kpr452 < 1U; i_kpr452++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr456 = 0U;
            for (; i_kpr456 < 1U; i_kpr456++) {
                uint32_t j_kpr457 = i_kpr456;
                vec_memcpy(elems_tile0 + (j_kpr457 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr457 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr458 = 0U;
            for (; i_kpr458 < 1U; i_kpr458++) {
                uint32_t j_kpr459 = i_kpr458;
                vec_memcpy(col_ind_tile0 + (j_kpr459 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr459 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr463 = 0U;
    for (; i_kpr463 < 8U; i_kpr463++) {
        uint32_t j_kpr464 = i_kpr463;
        uint32_t off = n_idx + j_kpr464 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr464 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_64x512x16_on
*/
static void
__hoisted_g_spmm_f32_64x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr465 = 0U;
        for (; i_kpr465 < 1U; i_kpr465++) {
            uint32_t j_kpr466 = i_kpr465;
            vec_memcpy(elems_tile0 + (j_kpr466 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr466 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr467 = 0U;
        for (; i_kpr467 < 1U; i_kpr467++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 64U;
        for (; nnz >= 64U; nnz -= 64U) {
            uint32_t off = ri_ + idx * 64U;
            __syncthreads();
            uint32_t i_kpr471 = 0U;
            for (; i_kpr471 < 1U; i_kpr471++) {
                uint32_t j_kpr472 = i_kpr471;
                vec_memcpy(elems_tile0 + (j_kpr472 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr472 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr473 = 0U;
            for (; i_kpr473 < 1U; i_kpr473++) {
                uint32_t j_kpr474 = i_kpr473;
                vec_memcpy(col_ind_tile0 + (j_kpr474 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr474 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr478 = 0U;
    for (; i_kpr478 < 8U; i_kpr478++) {
        uint32_t j_kpr479 = i_kpr478;
        uint32_t off = n_idx + j_kpr479 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr479 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x64x16
*/
static void
__hoisted_g_spmm_f32_128x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr480 = 0U;
        for (; i_kpr480 < 2U; i_kpr480++) {
            uint32_t j_kpr481 = i_kpr480;
            vec_memcpy(elems_tile0 + (j_kpr481 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr481 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr482 = 0U;
        for (; i_kpr482 < 2U; i_kpr482++) {
            uint32_t j_kpr483 = i_kpr482;
            vec_memcpy(col_ind_tile0 + (j_kpr483 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr483 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr484 = 0U;
        for (; i_kpr484 < to_; i_kpr484++)
            elems_tile0[i_kpr484 * 16U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr486 = 0U;
            for (; i_kpr486 < 2U; i_kpr486++) {
                uint32_t j_kpr487 = i_kpr486;
                vec_memcpy(elems_tile0 + (j_kpr487 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr487 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr488 = 0U;
            for (; i_kpr488 < 2U; i_kpr488++) {
                uint32_t j_kpr489 = i_kpr488;
                vec_memcpy(col_ind_tile0 + (j_kpr489 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr489 * 16U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr491 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr490 = 0U;
    for (; i_kpr490 < hi_kpr491; i_kpr490++) {
        uint32_t j_kpr492 = i_kpr490;
        elems_tile0[j_kpr492 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr492 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr492 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr492 * 16U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr493 = 0U;
    for (; i_kpr493 < 1U; i_kpr493++) {
        uint32_t j_kpr494 = i_kpr493;
        uint32_t off = n_idx + j_kpr494 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr494 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x64x16_on
*/
static void
__hoisted_g_spmm_f32_128x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr495 = 0U;
        for (; i_kpr495 < 2U; i_kpr495++) {
            uint32_t j_kpr496 = i_kpr495;
            vec_memcpy(elems_tile0 + (j_kpr496 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr496 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr497 = 0U;
        for (; i_kpr497 < 2U; i_kpr497++) {
            uint32_t j_kpr498 = i_kpr497;
            vec_memcpy(col_ind_tile0 + (j_kpr498 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr498 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr499 = 0U;
        for (; i_kpr499 < to_; i_kpr499++)
            elems_tile0[i_kpr499 * 16U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr501 = 0U;
            for (; i_kpr501 < 2U; i_kpr501++) {
                uint32_t j_kpr502 = i_kpr501;
                vec_memcpy(elems_tile0 + (j_kpr502 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr502 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr503 = 0U;
            for (; i_kpr503 < 2U; i_kpr503++) {
                uint32_t j_kpr504 = i_kpr503;
                vec_memcpy(col_ind_tile0 + (j_kpr504 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr504 * 16U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr506 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr505 = 0U;
    for (; i_kpr505 < hi_kpr506; i_kpr505++) {
        uint32_t j_kpr507 = i_kpr505;
        elems_tile0[j_kpr507 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr507 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr507 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr507 * 16U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr508 = 0U;
    for (; i_kpr508 < 1U; i_kpr508++) {
        uint32_t j_kpr509 = i_kpr508;
        uint32_t off = n_idx + j_kpr509 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr509 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x128x16
*/
static void
__hoisted_g_spmm_f32_128x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr510 = 0U;
        for (; i_kpr510 < 2U; i_kpr510++) {
            uint32_t j_kpr511 = i_kpr510;
            vec_memcpy(elems_tile0 + (j_kpr511 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr511 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr512 = 0U;
        for (; i_kpr512 < 2U; i_kpr512++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr516 = 0U;
            for (; i_kpr516 < 2U; i_kpr516++) {
                uint32_t j_kpr517 = i_kpr516;
                vec_memcpy(elems_tile0 + (j_kpr517 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr517 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr518 = 0U;
            for (; i_kpr518 < 2U; i_kpr518++) {
                uint32_t j_kpr519 = i_kpr518;
                vec_memcpy(col_ind_tile0 + (j_kpr519 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr519 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr523 = 0U;
    for (; i_kpr523 < 2U; i_kpr523++) {
        uint32_t j_kpr524 = i_kpr523;
        uint32_t off = n_idx + j_kpr524 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr524 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x128x16_on
*/
static void
__hoisted_g_spmm_f32_128x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr525 = 0U;
        for (; i_kpr525 < 2U; i_kpr525++) {
            uint32_t j_kpr526 = i_kpr525;
            vec_memcpy(elems_tile0 + (j_kpr526 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr526 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr527 = 0U;
        for (; i_kpr527 < 2U; i_kpr527++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr531 = 0U;
            for (; i_kpr531 < 2U; i_kpr531++) {
                uint32_t j_kpr532 = i_kpr531;
                vec_memcpy(elems_tile0 + (j_kpr532 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr532 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr533 = 0U;
            for (; i_kpr533 < 2U; i_kpr533++) {
                uint32_t j_kpr534 = i_kpr533;
                vec_memcpy(col_ind_tile0 + (j_kpr534 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr534 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr538 = 0U;
    for (; i_kpr538 < 2U; i_kpr538++) {
        uint32_t j_kpr539 = i_kpr538;
        uint32_t off = n_idx + j_kpr539 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr539 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x128x32
*/
static void
__hoisted_g_spmm_f32_128x128x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr540 = 0U;
        for (; i_kpr540 < 1U; i_kpr540++) {
            uint32_t j_kpr541 = i_kpr540;
            vec_memcpy(elems_tile0 + (j_kpr541 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr541 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr542 = 0U;
        for (; i_kpr542 < 1U; i_kpr542++) {
            uint32_t j_kpr543 = i_kpr542;
            vec_memcpy(col_ind_tile0 + (j_kpr543 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr543 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr544 = 0U;
        for (; i_kpr544 < to_; i_kpr544++)
            elems_tile0[i_kpr544 * 32U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr546 = 0U;
            for (; i_kpr546 < 1U; i_kpr546++) {
                uint32_t j_kpr547 = i_kpr546;
                vec_memcpy(elems_tile0 + (j_kpr547 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr547 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr548 = 0U;
            for (; i_kpr548 < 1U; i_kpr548++) {
                uint32_t j_kpr549 = i_kpr548;
                vec_memcpy(col_ind_tile0 + (j_kpr549 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr549 * 32U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr551 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr550 = 0U;
    for (; i_kpr550 < hi_kpr551; i_kpr550++) {
        uint32_t j_kpr552 = i_kpr550;
        elems_tile0[j_kpr552 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr552 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr552 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr552 * 32U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr553 = 0U;
    for (; i_kpr553 < 1U; i_kpr553++) {
        uint32_t j_kpr554 = i_kpr553;
        uint32_t off = n_idx + j_kpr554 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr554 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x128x32_on
*/
static void
__hoisted_g_spmm_f32_128x128x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr555 = 0U;
        for (; i_kpr555 < 1U; i_kpr555++) {
            uint32_t j_kpr556 = i_kpr555;
            vec_memcpy(elems_tile0 + (j_kpr556 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr556 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr557 = 0U;
        for (; i_kpr557 < 1U; i_kpr557++) {
            uint32_t j_kpr558 = i_kpr557;
            vec_memcpy(col_ind_tile0 + (j_kpr558 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr558 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr559 = 0U;
        for (; i_kpr559 < to_; i_kpr559++)
            elems_tile0[i_kpr559 * 32U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr561 = 0U;
            for (; i_kpr561 < 1U; i_kpr561++) {
                uint32_t j_kpr562 = i_kpr561;
                vec_memcpy(elems_tile0 + (j_kpr562 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr562 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr563 = 0U;
            for (; i_kpr563 < 1U; i_kpr563++) {
                uint32_t j_kpr564 = i_kpr563;
                vec_memcpy(col_ind_tile0 + (j_kpr564 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr564 * 32U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr566 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr565 = 0U;
    for (; i_kpr565 < hi_kpr566; i_kpr565++) {
        uint32_t j_kpr567 = i_kpr565;
        elems_tile0[j_kpr567 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr567 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr567 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr567 * 32U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr568 = 0U;
    for (; i_kpr568 < 1U; i_kpr568++) {
        uint32_t j_kpr569 = i_kpr568;
        uint32_t off = n_idx + j_kpr569 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr569 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x256x16
*/
static void
__hoisted_g_spmm_f32_128x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr570 = 0U;
        for (; i_kpr570 < 2U; i_kpr570++) {
            uint32_t j_kpr571 = i_kpr570;
            vec_memcpy(elems_tile0 + (j_kpr571 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr571 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr572 = 0U;
        for (; i_kpr572 < 2U; i_kpr572++) {
            uint32_t j_kpr573 = i_kpr572;
            vec_memcpy(col_ind_tile0 + (j_kpr573 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr573 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr574 = 0U;
        for (; i_kpr574 < to_; i_kpr574++)
            elems_tile0[i_kpr574 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr576 = 0U;
            for (; i_kpr576 < 2U; i_kpr576++) {
                uint32_t j_kpr577 = i_kpr576;
                vec_memcpy(elems_tile0 + (j_kpr577 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr577 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr578 = 0U;
            for (; i_kpr578 < 2U; i_kpr578++) {
                uint32_t j_kpr579 = i_kpr578;
                vec_memcpy(col_ind_tile0 + (j_kpr579 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr579 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr581 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr580 = 0U;
    for (; i_kpr580 < hi_kpr581; i_kpr580++) {
        uint32_t j_kpr582 = i_kpr580;
        elems_tile0[j_kpr582 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr582 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr582 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr582 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr583 = 0U;
    for (; i_kpr583 < 4U; i_kpr583++) {
        uint32_t j_kpr584 = i_kpr583;
        uint32_t off = n_idx + j_kpr584 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr584 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x256x16_on
*/
static void
__hoisted_g_spmm_f32_128x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr585 = 0U;
        for (; i_kpr585 < 2U; i_kpr585++) {
            uint32_t j_kpr586 = i_kpr585;
            vec_memcpy(elems_tile0 + (j_kpr586 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr586 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr587 = 0U;
        for (; i_kpr587 < 2U; i_kpr587++) {
            uint32_t j_kpr588 = i_kpr587;
            vec_memcpy(col_ind_tile0 + (j_kpr588 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr588 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr589 = 0U;
        for (; i_kpr589 < to_; i_kpr589++)
            elems_tile0[i_kpr589 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr591 = 0U;
            for (; i_kpr591 < 2U; i_kpr591++) {
                uint32_t j_kpr592 = i_kpr591;
                vec_memcpy(elems_tile0 + (j_kpr592 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr592 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr593 = 0U;
            for (; i_kpr593 < 2U; i_kpr593++) {
                uint32_t j_kpr594 = i_kpr593;
                vec_memcpy(col_ind_tile0 + (j_kpr594 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr594 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr596 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr595 = 0U;
    for (; i_kpr595 < hi_kpr596; i_kpr595++) {
        uint32_t j_kpr597 = i_kpr595;
        elems_tile0[j_kpr597 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr597 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr597 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr597 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr598 = 0U;
    for (; i_kpr598 < 4U; i_kpr598++) {
        uint32_t j_kpr599 = i_kpr598;
        uint32_t off = n_idx + j_kpr599 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr599 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x256x32
*/
static void
__hoisted_g_spmm_f32_128x256x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr600 = 0U;
        for (; i_kpr600 < 1U; i_kpr600++) {
            uint32_t j_kpr601 = i_kpr600;
            vec_memcpy(elems_tile0 + (j_kpr601 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr601 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr602 = 0U;
        for (; i_kpr602 < 1U; i_kpr602++) {
            uint32_t j_kpr603 = i_kpr602;
            vec_memcpy(col_ind_tile0 + (j_kpr603 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr603 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr604 = 0U;
        for (; i_kpr604 < to_; i_kpr604++)
            elems_tile0[i_kpr604 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr606 = 0U;
            for (; i_kpr606 < 1U; i_kpr606++) {
                uint32_t j_kpr607 = i_kpr606;
                vec_memcpy(elems_tile0 + (j_kpr607 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr607 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr608 = 0U;
            for (; i_kpr608 < 1U; i_kpr608++) {
                uint32_t j_kpr609 = i_kpr608;
                vec_memcpy(col_ind_tile0 + (j_kpr609 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr609 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr611 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr610 = 0U;
    for (; i_kpr610 < hi_kpr611; i_kpr610++) {
        uint32_t j_kpr612 = i_kpr610;
        elems_tile0[j_kpr612 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr612 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr612 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr612 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr613 = 0U;
    for (; i_kpr613 < 2U; i_kpr613++) {
        uint32_t j_kpr614 = i_kpr613;
        uint32_t off = n_idx + j_kpr614 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr614 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x256x32_on
*/
static void
__hoisted_g_spmm_f32_128x256x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr615 = 0U;
        for (; i_kpr615 < 1U; i_kpr615++) {
            uint32_t j_kpr616 = i_kpr615;
            vec_memcpy(elems_tile0 + (j_kpr616 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr616 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr617 = 0U;
        for (; i_kpr617 < 1U; i_kpr617++) {
            uint32_t j_kpr618 = i_kpr617;
            vec_memcpy(col_ind_tile0 + (j_kpr618 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr618 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr619 = 0U;
        for (; i_kpr619 < to_; i_kpr619++)
            elems_tile0[i_kpr619 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr621 = 0U;
            for (; i_kpr621 < 1U; i_kpr621++) {
                uint32_t j_kpr622 = i_kpr621;
                vec_memcpy(elems_tile0 + (j_kpr622 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr622 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr623 = 0U;
            for (; i_kpr623 < 1U; i_kpr623++) {
                uint32_t j_kpr624 = i_kpr623;
                vec_memcpy(col_ind_tile0 + (j_kpr624 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr624 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr626 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr625 = 0U;
    for (; i_kpr625 < hi_kpr626; i_kpr625++) {
        uint32_t j_kpr627 = i_kpr625;
        elems_tile0[j_kpr627 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr627 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr627 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr627 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr628 = 0U;
    for (; i_kpr628 < 2U; i_kpr628++) {
        uint32_t j_kpr629 = i_kpr628;
        uint32_t off = n_idx + j_kpr629 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr629 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x512x16
*/
static void
__hoisted_g_spmm_f32_128x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr630 = 0U;
        for (; i_kpr630 < 2U; i_kpr630++) {
            uint32_t j_kpr631 = i_kpr630;
            vec_memcpy(elems_tile0 + (j_kpr631 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr631 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr632 = 0U;
        for (; i_kpr632 < 2U; i_kpr632++) {
            uint32_t j_kpr633 = i_kpr632;
            vec_memcpy(col_ind_tile0 + (j_kpr633 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr633 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr634 = 0U;
        for (; i_kpr634 < to_; i_kpr634++)
            elems_tile0[i_kpr634 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr636 = 0U;
            for (; i_kpr636 < 2U; i_kpr636++) {
                uint32_t j_kpr637 = i_kpr636;
                vec_memcpy(elems_tile0 + (j_kpr637 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr637 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr638 = 0U;
            for (; i_kpr638 < 2U; i_kpr638++) {
                uint32_t j_kpr639 = i_kpr638;
                vec_memcpy(col_ind_tile0 + (j_kpr639 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr639 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr641 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr640 = 0U;
    for (; i_kpr640 < hi_kpr641; i_kpr640++) {
        uint32_t j_kpr642 = i_kpr640;
        elems_tile0[j_kpr642 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr642 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr642 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr642 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr643 = 0U;
    for (; i_kpr643 < 8U; i_kpr643++) {
        uint32_t j_kpr644 = i_kpr643;
        uint32_t off = n_idx + j_kpr644 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr644 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_128x512x16_on
*/
static void
__hoisted_g_spmm_f32_128x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr645 = 0U;
        for (; i_kpr645 < 2U; i_kpr645++) {
            uint32_t j_kpr646 = i_kpr645;
            vec_memcpy(elems_tile0 + (j_kpr646 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr646 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr647 = 0U;
        for (; i_kpr647 < 2U; i_kpr647++) {
            uint32_t j_kpr648 = i_kpr647;
            vec_memcpy(col_ind_tile0 + (j_kpr648 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr648 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr649 = 0U;
        for (; i_kpr649 < to_; i_kpr649++)
            elems_tile0[i_kpr649 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr651 = 0U;
            for (; i_kpr651 < 2U; i_kpr651++) {
                uint32_t j_kpr652 = i_kpr651;
                vec_memcpy(elems_tile0 + (j_kpr652 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr652 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr653 = 0U;
            for (; i_kpr653 < 2U; i_kpr653++) {
                uint32_t j_kpr654 = i_kpr653;
                vec_memcpy(col_ind_tile0 + (j_kpr654 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr654 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr656 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr655 = 0U;
    for (; i_kpr655 < hi_kpr656; i_kpr655++) {
        uint32_t j_kpr657 = i_kpr655;
        elems_tile0[j_kpr657 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr657 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr657 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr657 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr658 = 0U;
    for (; i_kpr658 < 8U; i_kpr658++) {
        uint32_t j_kpr659 = i_kpr658;
        uint32_t off = n_idx + j_kpr659 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr659 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x512x32
*/
static void
__hoisted_g_spmm_f32_128x512x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr660 = 0U;
        for (; i_kpr660 < 1U; i_kpr660++) {
            uint32_t j_kpr661 = i_kpr660;
            vec_memcpy(elems_tile0 + (j_kpr661 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr661 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr662 = 0U;
        for (; i_kpr662 < 1U; i_kpr662++) {
            uint32_t j_kpr663 = i_kpr662;
            vec_memcpy(col_ind_tile0 + (j_kpr663 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr663 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr664 = 0U;
        for (; i_kpr664 < to_; i_kpr664++)
            elems_tile0[i_kpr664 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr666 = 0U;
            for (; i_kpr666 < 1U; i_kpr666++) {
                uint32_t j_kpr667 = i_kpr666;
                vec_memcpy(elems_tile0 + (j_kpr667 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr667 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr668 = 0U;
            for (; i_kpr668 < 1U; i_kpr668++) {
                uint32_t j_kpr669 = i_kpr668;
                vec_memcpy(col_ind_tile0 + (j_kpr669 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr669 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr671 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr670 = 0U;
    for (; i_kpr670 < hi_kpr671; i_kpr670++) {
        uint32_t j_kpr672 = i_kpr670;
        elems_tile0[j_kpr672 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr672 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr672 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr672 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr673 = 0U;
    for (; i_kpr673 < 4U; i_kpr673++) {
        uint32_t j_kpr674 = i_kpr673;
        uint32_t off = n_idx + j_kpr674 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr674 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_128x512x32_on
*/
static void
__hoisted_g_spmm_f32_128x512x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr675 = 0U;
        for (; i_kpr675 < 1U; i_kpr675++) {
            uint32_t j_kpr676 = i_kpr675;
            vec_memcpy(elems_tile0 + (j_kpr676 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr676 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr677 = 0U;
        for (; i_kpr677 < 1U; i_kpr677++) {
            uint32_t j_kpr678 = i_kpr677;
            vec_memcpy(col_ind_tile0 + (j_kpr678 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr678 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr679 = 0U;
        for (; i_kpr679 < to_; i_kpr679++)
            elems_tile0[i_kpr679 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 128U;
        for (; nnz >= 128U; nnz -= 128U) {
            uint32_t off = ri_ + idx * 128U;
            __syncthreads();
            uint32_t i_kpr681 = 0U;
            for (; i_kpr681 < 1U; i_kpr681++) {
                uint32_t j_kpr682 = i_kpr681;
                vec_memcpy(elems_tile0 + (j_kpr682 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr682 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr683 = 0U;
            for (; i_kpr683 < 1U; i_kpr683++) {
                uint32_t j_kpr684 = i_kpr683;
                vec_memcpy(col_ind_tile0 + (j_kpr684 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr684 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr686 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr685 = 0U;
    for (; i_kpr685 < hi_kpr686; i_kpr685++) {
        uint32_t j_kpr687 = i_kpr685;
        elems_tile0[j_kpr687 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr687 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr687 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr687 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr688 = 0U;
    for (; i_kpr688 < 4U; i_kpr688++) {
        uint32_t j_kpr689 = i_kpr688;
        uint32_t off = n_idx + j_kpr689 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr689 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x64x16
*/
static void
__hoisted_g_spmm_f32_256x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr703 = 0U;
    for (; i_kpr703 < 1U; i_kpr703++) {
        uint32_t j_kpr704 = i_kpr703;
        uint32_t off = n_idx + j_kpr704 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr704 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x64x16_on
*/
static void
__hoisted_g_spmm_f32_256x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr718 = 0U;
    for (; i_kpr718 < 1U; i_kpr718++) {
        uint32_t j_kpr719 = i_kpr718;
        uint32_t off = n_idx + j_kpr719 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr719 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x128x16
*/
static void
__hoisted_g_spmm_f32_256x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr720 = 0U;
        for (; i_kpr720 < 4U; i_kpr720++) {
            uint32_t j_kpr721 = i_kpr720;
            vec_memcpy(elems_tile0 + (j_kpr721 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr721 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr722 = 0U;
        for (; i_kpr722 < 4U; i_kpr722++) {
            uint32_t j_kpr723 = i_kpr722;
            vec_memcpy(col_ind_tile0 + (j_kpr723 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr723 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr724 = 0U;
        for (; i_kpr724 < to_; i_kpr724++)
            elems_tile0[i_kpr724 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; i_kpr726 < 4U; i_kpr726++) {
                uint32_t j_kpr727 = i_kpr726;
                vec_memcpy(elems_tile0 + (j_kpr727 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr727 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr728 = 0U;
            for (; i_kpr728 < 4U; i_kpr728++) {
                uint32_t j_kpr729 = i_kpr728;
                vec_memcpy(col_ind_tile0 + (j_kpr729 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr729 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr731 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr730 = 0U;
    for (; i_kpr730 < hi_kpr731; i_kpr730++) {
        uint32_t j_kpr732 = i_kpr730;
        elems_tile0[j_kpr732 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr732 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr732 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr732 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr733 = 0U;
    for (; i_kpr733 < 2U; i_kpr733++) {
        uint32_t j_kpr734 = i_kpr733;
        uint32_t off = n_idx + j_kpr734 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr734 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x128x16_on
*/
static void
__hoisted_g_spmm_f32_256x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr735 = 0U;
        for (; i_kpr735 < 4U; i_kpr735++) {
            uint32_t j_kpr736 = i_kpr735;
            vec_memcpy(elems_tile0 + (j_kpr736 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr736 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr737 = 0U;
        for (; i_kpr737 < 4U; i_kpr737++) {
            uint32_t j_kpr738 = i_kpr737;
            vec_memcpy(col_ind_tile0 + (j_kpr738 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr738 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr739 = 0U;
        for (; i_kpr739 < to_; i_kpr739++)
            elems_tile0[i_kpr739 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; i_kpr741 < 4U; i_kpr741++) {
                uint32_t j_kpr742 = i_kpr741;
                vec_memcpy(elems_tile0 + (j_kpr742 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr742 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr743 = 0U;
            for (; i_kpr743 < 4U; i_kpr743++) {
                uint32_t j_kpr744 = i_kpr743;
                vec_memcpy(col_ind_tile0 + (j_kpr744 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr744 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr746 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr745 = 0U;
    for (; i_kpr745 < hi_kpr746; i_kpr745++) {
        uint32_t j_kpr747 = i_kpr745;
        elems_tile0[j_kpr747 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr747 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr747 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr747 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr748 = 0U;
    for (; i_kpr748 < 2U; i_kpr748++) {
        uint32_t j_kpr749 = i_kpr748;
        uint32_t off = n_idx + j_kpr749 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr749 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x128x32
*/
static void
__hoisted_g_spmm_f32_256x128x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr750 = 0U;
        for (; i_kpr750 < 2U; i_kpr750++) {
            uint32_t j_kpr751 = i_kpr750;
            vec_memcpy(elems_tile0 + (j_kpr751 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr751 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr752 = 0U;
        for (; i_kpr752 < 2U; i_kpr752++) {
            uint32_t j_kpr753 = i_kpr752;
            vec_memcpy(col_ind_tile0 + (j_kpr753 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr753 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr754 = 0U;
        for (; i_kpr754 < to_; i_kpr754++)
            elems_tile0[i_kpr754 * 32U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr756 = 0U;
            for (; i_kpr756 < 2U; i_kpr756++) {
                uint32_t j_kpr757 = i_kpr756;
                vec_memcpy(elems_tile0 + (j_kpr757 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr757 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr758 = 0U;
            for (; i_kpr758 < 2U; i_kpr758++) {
                uint32_t j_kpr759 = i_kpr758;
                vec_memcpy(col_ind_tile0 + (j_kpr759 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr759 * 32U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr761 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr760 = 0U;
    for (; i_kpr760 < hi_kpr761; i_kpr760++) {
        uint32_t j_kpr762 = i_kpr760;
        elems_tile0[j_kpr762 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr762 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr762 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr762 * 32U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr763 = 0U;
    for (; i_kpr763 < 1U; i_kpr763++) {
        uint32_t j_kpr764 = i_kpr763;
        uint32_t off = n_idx + j_kpr764 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr764 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x128x32_on
*/
static void
__hoisted_g_spmm_f32_256x128x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr765 = 0U;
        for (; i_kpr765 < 2U; i_kpr765++) {
            uint32_t j_kpr766 = i_kpr765;
            vec_memcpy(elems_tile0 + (j_kpr766 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr766 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr767 = 0U;
        for (; i_kpr767 < 2U; i_kpr767++) {
            uint32_t j_kpr768 = i_kpr767;
            vec_memcpy(col_ind_tile0 + (j_kpr768 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr768 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr769 = 0U;
        for (; i_kpr769 < to_; i_kpr769++)
            elems_tile0[i_kpr769 * 32U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr771 = 0U;
            for (; i_kpr771 < 2U; i_kpr771++) {
                uint32_t j_kpr772 = i_kpr771;
                vec_memcpy(elems_tile0 + (j_kpr772 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr772 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr773 = 0U;
            for (; i_kpr773 < 2U; i_kpr773++) {
                uint32_t j_kpr774 = i_kpr773;
                vec_memcpy(col_ind_tile0 + (j_kpr774 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr774 * 32U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr776 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr775 = 0U;
    for (; i_kpr775 < hi_kpr776; i_kpr775++) {
        uint32_t j_kpr777 = i_kpr775;
        elems_tile0[j_kpr777 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr777 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr777 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr777 * 32U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr778 = 0U;
    for (; i_kpr778 < 1U; i_kpr778++) {
        uint32_t j_kpr779 = i_kpr778;
        uint32_t off = n_idx + j_kpr779 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr779 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x256x16
*/
static void
__hoisted_g_spmm_f32_256x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr780 = 0U;
        for (; i_kpr780 < 4U; i_kpr780++) {
            uint32_t j_kpr781 = i_kpr780;
            vec_memcpy(elems_tile0 + (j_kpr781 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr781 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr782 = 0U;
        for (; i_kpr782 < 4U; i_kpr782++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr786 = 0U;
            for (; i_kpr786 < 4U; i_kpr786++) {
                uint32_t j_kpr787 = i_kpr786;
                vec_memcpy(elems_tile0 + (j_kpr787 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr787 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr788 = 0U;
            for (; i_kpr788 < 4U; i_kpr788++) {
                uint32_t j_kpr789 = i_kpr788;
                vec_memcpy(col_ind_tile0 + (j_kpr789 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr789 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr793 = 0U;
    for (; i_kpr793 < 4U; i_kpr793++) {
        uint32_t j_kpr794 = i_kpr793;
        uint32_t off = n_idx + j_kpr794 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr794 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x256x16_on
*/
static void
__hoisted_g_spmm_f32_256x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr795 = 0U;
        for (; i_kpr795 < 4U; i_kpr795++) {
            uint32_t j_kpr796 = i_kpr795;
            vec_memcpy(elems_tile0 + (j_kpr796 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr796 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr797 = 0U;
        for (; i_kpr797 < 4U; i_kpr797++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr801 = 0U;
            for (; i_kpr801 < 4U; i_kpr801++) {
                uint32_t j_kpr802 = i_kpr801;
                vec_memcpy(elems_tile0 + (j_kpr802 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr802 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr803 = 0U;
            for (; i_kpr803 < 4U; i_kpr803++) {
                uint32_t j_kpr804 = i_kpr803;
                vec_memcpy(col_ind_tile0 + (j_kpr804 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr804 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr808 = 0U;
    for (; i_kpr808 < 4U; i_kpr808++) {
        uint32_t j_kpr809 = i_kpr808;
        uint32_t off = n_idx + j_kpr809 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr809 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x256x32
*/
static void
__hoisted_g_spmm_f32_256x256x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr810 = 0U;
        for (; i_kpr810 < 2U; i_kpr810++) {
            uint32_t j_kpr811 = i_kpr810;
            vec_memcpy(elems_tile0 + (j_kpr811 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr811 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr812 = 0U;
        for (; i_kpr812 < 2U; i_kpr812++) {
            uint32_t j_kpr813 = i_kpr812;
            vec_memcpy(col_ind_tile0 + (j_kpr813 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr813 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr814 = 0U;
        for (; i_kpr814 < to_; i_kpr814++)
            elems_tile0[i_kpr814 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr816 = 0U;
            for (; i_kpr816 < 2U; i_kpr816++) {
                uint32_t j_kpr817 = i_kpr816;
                vec_memcpy(elems_tile0 + (j_kpr817 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr817 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr818 = 0U;
            for (; i_kpr818 < 2U; i_kpr818++) {
                uint32_t j_kpr819 = i_kpr818;
                vec_memcpy(col_ind_tile0 + (j_kpr819 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr819 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr821 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr820 = 0U;
    for (; i_kpr820 < hi_kpr821; i_kpr820++) {
        uint32_t j_kpr822 = i_kpr820;
        elems_tile0[j_kpr822 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr822 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr822 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr822 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr823 = 0U;
    for (; i_kpr823 < 2U; i_kpr823++) {
        uint32_t j_kpr824 = i_kpr823;
        uint32_t off = n_idx + j_kpr824 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr824 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x256x32_on
*/
static void
__hoisted_g_spmm_f32_256x256x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr825 = 0U;
        for (; i_kpr825 < 2U; i_kpr825++) {
            uint32_t j_kpr826 = i_kpr825;
            vec_memcpy(elems_tile0 + (j_kpr826 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr826 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr827 = 0U;
        for (; i_kpr827 < 2U; i_kpr827++) {
            uint32_t j_kpr828 = i_kpr827;
            vec_memcpy(col_ind_tile0 + (j_kpr828 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr828 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr829 = 0U;
        for (; i_kpr829 < to_; i_kpr829++)
            elems_tile0[i_kpr829 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr831 = 0U;
            for (; i_kpr831 < 2U; i_kpr831++) {
                uint32_t j_kpr832 = i_kpr831;
                vec_memcpy(elems_tile0 + (j_kpr832 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr832 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr833 = 0U;
            for (; i_kpr833 < 2U; i_kpr833++) {
                uint32_t j_kpr834 = i_kpr833;
                vec_memcpy(col_ind_tile0 + (j_kpr834 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr834 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr836 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr835 = 0U;
    for (; i_kpr835 < hi_kpr836; i_kpr835++) {
        uint32_t j_kpr837 = i_kpr835;
        elems_tile0[j_kpr837 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr837 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr837 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr837 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr838 = 0U;
    for (; i_kpr838 < 2U; i_kpr838++) {
        uint32_t j_kpr839 = i_kpr838;
        uint32_t off = n_idx + j_kpr839 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr839 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x256x64
*/
static void
__hoisted_g_spmm_f32_256x256x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr840 = 0U;
        for (; i_kpr840 < 1U; i_kpr840++) {
            uint32_t j_kpr841 = i_kpr840;
            vec_memcpy(elems_tile0 + (j_kpr841 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr841 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr842 = 0U;
        for (; i_kpr842 < 1U; i_kpr842++) {
            uint32_t j_kpr843 = i_kpr842;
            vec_memcpy(col_ind_tile0 + (j_kpr843 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr843 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr844 = 0U;
        for (; i_kpr844 < to_; i_kpr844++)
            elems_tile0[i_kpr844 * 64U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr846 = 0U;
            for (; i_kpr846 < 1U; i_kpr846++) {
                uint32_t j_kpr847 = i_kpr846;
                vec_memcpy(elems_tile0 + (j_kpr847 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr847 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr848 = 0U;
            for (; i_kpr848 < 1U; i_kpr848++) {
                uint32_t j_kpr849 = i_kpr848;
                vec_memcpy(col_ind_tile0 + (j_kpr849 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr849 * 64U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr851 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr850 = 0U;
    for (; i_kpr850 < hi_kpr851; i_kpr850++) {
        uint32_t j_kpr852 = i_kpr850;
        elems_tile0[j_kpr852 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr852 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr852 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr852 * 64U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr853 = 0U;
    for (; i_kpr853 < 1U; i_kpr853++) {
        uint32_t j_kpr854 = i_kpr853;
        uint32_t off = n_idx + j_kpr854 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr854 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x256x64_on
*/
static void
__hoisted_g_spmm_f32_256x256x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr855 = 0U;
        for (; i_kpr855 < 1U; i_kpr855++) {
            uint32_t j_kpr856 = i_kpr855;
            vec_memcpy(elems_tile0 + (j_kpr856 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr856 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr857 = 0U;
        for (; i_kpr857 < 1U; i_kpr857++) {
            uint32_t j_kpr858 = i_kpr857;
            vec_memcpy(col_ind_tile0 + (j_kpr858 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr858 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr859 = 0U;
        for (; i_kpr859 < to_; i_kpr859++)
            elems_tile0[i_kpr859 * 64U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr861 = 0U;
            for (; i_kpr861 < 1U; i_kpr861++) {
                uint32_t j_kpr862 = i_kpr861;
                vec_memcpy(elems_tile0 + (j_kpr862 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr862 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr863 = 0U;
            for (; i_kpr863 < 1U; i_kpr863++) {
                uint32_t j_kpr864 = i_kpr863;
                vec_memcpy(col_ind_tile0 + (j_kpr864 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr864 * 64U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr866 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr865 = 0U;
    for (; i_kpr865 < hi_kpr866; i_kpr865++) {
        uint32_t j_kpr867 = i_kpr865;
        elems_tile0[j_kpr867 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr867 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr867 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr867 * 64U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr868 = 0U;
    for (; i_kpr868 < 1U; i_kpr868++) {
        uint32_t j_kpr869 = i_kpr868;
        uint32_t off = n_idx + j_kpr869 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr869 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x512x16
*/
static void
__hoisted_g_spmm_f32_256x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr870 = 0U;
        for (; i_kpr870 < 4U; i_kpr870++) {
            uint32_t j_kpr871 = i_kpr870;
            vec_memcpy(elems_tile0 + (j_kpr871 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr871 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr872 = 0U;
        for (; i_kpr872 < 4U; i_kpr872++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr876 = 0U;
            for (; i_kpr876 < 4U; i_kpr876++) {
                uint32_t j_kpr877 = i_kpr876;
                vec_memcpy(elems_tile0 + (j_kpr877 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr877 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr878 = 0U;
            for (; i_kpr878 < 4U; i_kpr878++) {
                uint32_t j_kpr879 = i_kpr878;
                vec_memcpy(col_ind_tile0 + (j_kpr879 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr879 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr883 = 0U;
    for (; i_kpr883 < 8U; i_kpr883++) {
        uint32_t j_kpr884 = i_kpr883;
        uint32_t off = n_idx + j_kpr884 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr884 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_256x512x16_on
*/
static void
__hoisted_g_spmm_f32_256x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr885 = 0U;
        for (; i_kpr885 < 4U; i_kpr885++) {
            uint32_t j_kpr886 = i_kpr885;
            vec_memcpy(elems_tile0 + (j_kpr886 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr886 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr887 = 0U;
        for (; i_kpr887 < 4U; i_kpr887++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr891 = 0U;
            for (; i_kpr891 < 4U; i_kpr891++) {
                uint32_t j_kpr892 = i_kpr891;
                vec_memcpy(elems_tile0 + (j_kpr892 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr892 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr893 = 0U;
            for (; i_kpr893 < 4U; i_kpr893++) {
                uint32_t j_kpr894 = i_kpr893;
                vec_memcpy(col_ind_tile0 + (j_kpr894 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr894 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr898 = 0U;
    for (; i_kpr898 < 8U; i_kpr898++) {
        uint32_t j_kpr899 = i_kpr898;
        uint32_t off = n_idx + j_kpr899 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr899 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x512x32
*/
static void
__hoisted_g_spmm_f32_256x512x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr900 = 0U;
        for (; i_kpr900 < 2U; i_kpr900++) {
            uint32_t j_kpr901 = i_kpr900;
            vec_memcpy(elems_tile0 + (j_kpr901 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr901 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr902 = 0U;
        for (; i_kpr902 < 2U; i_kpr902++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr906 = 0U;
            for (; i_kpr906 < 2U; i_kpr906++) {
                uint32_t j_kpr907 = i_kpr906;
                vec_memcpy(elems_tile0 + (j_kpr907 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr907 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr908 = 0U;
            for (; i_kpr908 < 2U; i_kpr908++) {
                uint32_t j_kpr909 = i_kpr908;
                vec_memcpy(col_ind_tile0 + (j_kpr909 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr909 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr913 = 0U;
    for (; i_kpr913 < 4U; i_kpr913++) {
        uint32_t j_kpr914 = i_kpr913;
        uint32_t off = n_idx + j_kpr914 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr914 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_256x512x32_on
*/
static void
__hoisted_g_spmm_f32_256x512x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr915 = 0U;
        for (; i_kpr915 < 2U; i_kpr915++) {
            uint32_t j_kpr916 = i_kpr915;
            vec_memcpy(elems_tile0 + (j_kpr916 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr916 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr917 = 0U;
        for (; i_kpr917 < 2U; i_kpr917++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr921 = 0U;
            for (; i_kpr921 < 2U; i_kpr921++) {
                uint32_t j_kpr922 = i_kpr921;
                vec_memcpy(elems_tile0 + (j_kpr922 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr922 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr923 = 0U;
            for (; i_kpr923 < 2U; i_kpr923++) {
                uint32_t j_kpr924 = i_kpr923;
                vec_memcpy(col_ind_tile0 + (j_kpr924 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr924 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr928 = 0U;
    for (; i_kpr928 < 4U; i_kpr928++) {
        uint32_t j_kpr929 = i_kpr928;
        uint32_t off = n_idx + j_kpr929 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr929 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x512x64
*/
static void
__hoisted_g_spmm_f32_256x512x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr930 = 0U;
        for (; i_kpr930 < 1U; i_kpr930++) {
            uint32_t j_kpr931 = i_kpr930;
            vec_memcpy(elems_tile0 + (j_kpr931 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr931 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr932 = 0U;
        for (; i_kpr932 < 1U; i_kpr932++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr936 = 0U;
            for (; i_kpr936 < 1U; i_kpr936++) {
                uint32_t j_kpr937 = i_kpr936;
                vec_memcpy(elems_tile0 + (j_kpr937 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr937 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr938 = 0U;
            for (; i_kpr938 < 1U; i_kpr938++) {
                uint32_t j_kpr939 = i_kpr938;
                vec_memcpy(col_ind_tile0 + (j_kpr939 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr939 * 64U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr943 = 0U;
    for (; i_kpr943 < 2U; i_kpr943++) {
        uint32_t j_kpr944 = i_kpr943;
        uint32_t off = n_idx + j_kpr944 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr944 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_256x512x64_on
*/
static void
__hoisted_g_spmm_f32_256x512x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr945 = 0U;
        for (; i_kpr945 < 1U; i_kpr945++) {
            uint32_t j_kpr946 = i_kpr945;
            vec_memcpy(elems_tile0 + (j_kpr946 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr946 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr947 = 0U;
        for (; i_kpr947 < 1U; i_kpr947++) {
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
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 256U;
        for (; nnz >= 256U; nnz -= 256U) {
            uint32_t off = ri_ + idx * 256U;
            __syncthreads();
            uint32_t i_kpr951 = 0U;
            for (; i_kpr951 < 1U; i_kpr951++) {
                uint32_t j_kpr952 = i_kpr951;
                vec_memcpy(elems_tile0 + (j_kpr952 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr952 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr953 = 0U;
            for (; i_kpr953 < 1U; i_kpr953++) {
                uint32_t j_kpr954 = i_kpr953;
                vec_memcpy(col_ind_tile0 + (j_kpr954 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr954 * 64U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 2U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 64U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr958 = 0U;
    for (; i_kpr958 < 2U; i_kpr958++) {
        uint32_t j_kpr959 = i_kpr958;
        uint32_t off = n_idx + j_kpr959 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr959 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x64x16
*/
static void
__hoisted_g_spmm_f32_512x64x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr973 = 0U;
    for (; i_kpr973 < 1U; i_kpr973++) {
        uint32_t j_kpr974 = i_kpr973;
        uint32_t off = n_idx + j_kpr974 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr974 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x64x16_on
*/
static void
__hoisted_g_spmm_f32_512x64x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        float lchunk[4U];
                        memset(lchunk, 0U, 4U * sizeof(float));
                        vec_memcpy(lchunk,
                            gB + (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                float lchunk[4U];
                memset(lchunk, 0U, 4U * sizeof(float));
                vec_memcpy(
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr988 = 0U;
    for (; i_kpr988 < 1U; i_kpr988++) {
        uint32_t j_kpr989 = i_kpr988;
        uint32_t off = n_idx + j_kpr989 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr989 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x128x16
*/
static void
__hoisted_g_spmm_f32_512x128x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr990 = 0U;
        for (; i_kpr990 < 8U; i_kpr990++) {
            uint32_t j_kpr991 = i_kpr990;
            vec_memcpy(elems_tile0 + (j_kpr991 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr991 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr992 = 0U;
        for (; i_kpr992 < 8U; i_kpr992++) {
            uint32_t j_kpr993 = i_kpr992;
            vec_memcpy(col_ind_tile0 + (j_kpr993 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr993 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr994 = 0U;
        for (; i_kpr994 < to_; i_kpr994++)
            elems_tile0[i_kpr994 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; i_kpr996 < 8U; i_kpr996++) {
                uint32_t j_kpr997 = i_kpr996;
                vec_memcpy(elems_tile0 + (j_kpr997 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr997 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr998 = 0U;
            for (; i_kpr998 < 8U; i_kpr998++) {
                uint32_t j_kpr999 = i_kpr998;
                vec_memcpy(col_ind_tile0 + (j_kpr999 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr999 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1001 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1000 = 0U;
    for (; i_kpr1000 < hi_kpr1001; i_kpr1000++) {
        uint32_t j_kpr1002 = i_kpr1000;
        elems_tile0[j_kpr1002 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1002 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1002 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1002 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1003 = 0U;
    for (; i_kpr1003 < 2U; i_kpr1003++) {
        uint32_t j_kpr1004 = i_kpr1003;
        uint32_t off = n_idx + j_kpr1004 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1004 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x128x16_on
*/
static void
__hoisted_g_spmm_f32_512x128x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1005 = 0U;
        for (; i_kpr1005 < 8U; i_kpr1005++) {
            uint32_t j_kpr1006 = i_kpr1005;
            vec_memcpy(elems_tile0 + (j_kpr1006 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1006 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1007 = 0U;
        for (; i_kpr1007 < 8U; i_kpr1007++) {
            uint32_t j_kpr1008 = i_kpr1007;
            vec_memcpy(col_ind_tile0 + (j_kpr1008 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1008 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1009 = 0U;
        for (; i_kpr1009 < to_; i_kpr1009++)
            elems_tile0[i_kpr1009 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; i_kpr1011 < 8U; i_kpr1011++) {
                uint32_t j_kpr1012 = i_kpr1011;
                vec_memcpy(elems_tile0 + (j_kpr1012 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1012 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1013 = 0U;
            for (; i_kpr1013 < 8U; i_kpr1013++) {
                uint32_t j_kpr1014 = i_kpr1013;
                vec_memcpy(col_ind_tile0 + (j_kpr1014 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1014 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1016 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1015 = 0U;
    for (; i_kpr1015 < hi_kpr1016; i_kpr1015++) {
        uint32_t j_kpr1017 = i_kpr1015;
        elems_tile0[j_kpr1017 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1017 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1017 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1017 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1018 = 0U;
    for (; i_kpr1018 < 2U; i_kpr1018++) {
        uint32_t j_kpr1019 = i_kpr1018;
        uint32_t off = n_idx + j_kpr1019 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1019 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x128x32
*/
static void
__hoisted_g_spmm_f32_512x128x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1020 = 0U;
        for (; i_kpr1020 < 4U; i_kpr1020++) {
            uint32_t j_kpr1021 = i_kpr1020;
            vec_memcpy(elems_tile0 + (j_kpr1021 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1021 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1022 = 0U;
        for (; i_kpr1022 < 4U; i_kpr1022++) {
            uint32_t j_kpr1023 = i_kpr1022;
            vec_memcpy(col_ind_tile0 + (j_kpr1023 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1023 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1024 = 0U;
        for (; i_kpr1024 < to_; i_kpr1024++)
            elems_tile0[i_kpr1024 * 32U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1026 = 0U;
            for (; i_kpr1026 < 4U; i_kpr1026++) {
                uint32_t j_kpr1027 = i_kpr1026;
                vec_memcpy(elems_tile0 + (j_kpr1027 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1027 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1028 = 0U;
            for (; i_kpr1028 < 4U; i_kpr1028++) {
                uint32_t j_kpr1029 = i_kpr1028;
                vec_memcpy(col_ind_tile0 + (j_kpr1029 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1029 * 32U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1031 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1030 = 0U;
    for (; i_kpr1030 < hi_kpr1031; i_kpr1030++) {
        uint32_t j_kpr1032 = i_kpr1030;
        elems_tile0[j_kpr1032 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1032 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1032 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1032 * 32U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1033 = 0U;
    for (; i_kpr1033 < 1U; i_kpr1033++) {
        uint32_t j_kpr1034 = i_kpr1033;
        uint32_t off = n_idx + j_kpr1034 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1034 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x128x32_on
*/
static void
__hoisted_g_spmm_f32_512x128x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1035 = 0U;
        for (; i_kpr1035 < 4U; i_kpr1035++) {
            uint32_t j_kpr1036 = i_kpr1035;
            vec_memcpy(elems_tile0 + (j_kpr1036 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1036 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1037 = 0U;
        for (; i_kpr1037 < 4U; i_kpr1037++) {
            uint32_t j_kpr1038 = i_kpr1037;
            vec_memcpy(col_ind_tile0 + (j_kpr1038 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1038 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1039 = 0U;
        for (; i_kpr1039 < to_; i_kpr1039++)
            elems_tile0[i_kpr1039 * 32U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1041 = 0U;
            for (; i_kpr1041 < 4U; i_kpr1041++) {
                uint32_t j_kpr1042 = i_kpr1041;
                vec_memcpy(elems_tile0 + (j_kpr1042 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1042 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1043 = 0U;
            for (; i_kpr1043 < 4U; i_kpr1043++) {
                uint32_t j_kpr1044 = i_kpr1043;
                vec_memcpy(col_ind_tile0 + (j_kpr1044 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1044 * 32U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1046 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1045 = 0U;
    for (; i_kpr1045 < hi_kpr1046; i_kpr1045++) {
        uint32_t j_kpr1047 = i_kpr1045;
        elems_tile0[j_kpr1047 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1047 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1047 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1047 * 32U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1048 = 0U;
    for (; i_kpr1048 < 1U; i_kpr1048++) {
        uint32_t j_kpr1049 = i_kpr1048;
        uint32_t off = n_idx + j_kpr1049 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1049 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x256x16
*/
static void
__hoisted_g_spmm_f32_512x256x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1050 = 0U;
        for (; i_kpr1050 < 8U; i_kpr1050++) {
            uint32_t j_kpr1051 = i_kpr1050;
            vec_memcpy(elems_tile0 + (j_kpr1051 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1051 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1052 = 0U;
        for (; i_kpr1052 < 8U; i_kpr1052++) {
            uint32_t j_kpr1053 = i_kpr1052;
            vec_memcpy(col_ind_tile0 + (j_kpr1053 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1053 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1054 = 0U;
        for (; i_kpr1054 < to_; i_kpr1054++)
            elems_tile0[i_kpr1054 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; i_kpr1056 < 8U; i_kpr1056++) {
                uint32_t j_kpr1057 = i_kpr1056;
                vec_memcpy(elems_tile0 + (j_kpr1057 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1057 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1058 = 0U;
            for (; i_kpr1058 < 8U; i_kpr1058++) {
                uint32_t j_kpr1059 = i_kpr1058;
                vec_memcpy(col_ind_tile0 + (j_kpr1059 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1059 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1061 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1060 = 0U;
    for (; i_kpr1060 < hi_kpr1061; i_kpr1060++) {
        uint32_t j_kpr1062 = i_kpr1060;
        elems_tile0[j_kpr1062 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1062 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1062 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1062 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1063 = 0U;
    for (; i_kpr1063 < 4U; i_kpr1063++) {
        uint32_t j_kpr1064 = i_kpr1063;
        uint32_t off = n_idx + j_kpr1064 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1064 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x256x16_on
*/
static void
__hoisted_g_spmm_f32_512x256x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1065 = 0U;
        for (; i_kpr1065 < 8U; i_kpr1065++) {
            uint32_t j_kpr1066 = i_kpr1065;
            vec_memcpy(elems_tile0 + (j_kpr1066 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1066 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1067 = 0U;
        for (; i_kpr1067 < 8U; i_kpr1067++) {
            uint32_t j_kpr1068 = i_kpr1067;
            vec_memcpy(col_ind_tile0 + (j_kpr1068 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1068 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1069 = 0U;
        for (; i_kpr1069 < to_; i_kpr1069++)
            elems_tile0[i_kpr1069 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
            for (; i_kpr1071 < 8U; i_kpr1071++) {
                uint32_t j_kpr1072 = i_kpr1071;
                vec_memcpy(elems_tile0 + (j_kpr1072 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1072 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1073 = 0U;
            for (; i_kpr1073 < 8U; i_kpr1073++) {
                uint32_t j_kpr1074 = i_kpr1073;
                vec_memcpy(col_ind_tile0 + (j_kpr1074 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1074 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1076 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1075 = 0U;
    for (; i_kpr1075 < hi_kpr1076; i_kpr1075++) {
        uint32_t j_kpr1077 = i_kpr1075;
        elems_tile0[j_kpr1077 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1077 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1077 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1077 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1078 = 0U;
    for (; i_kpr1078 < 4U; i_kpr1078++) {
        uint32_t j_kpr1079 = i_kpr1078;
        uint32_t off = n_idx + j_kpr1079 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1079 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x256x32
*/
static void
__hoisted_g_spmm_f32_512x256x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1080 = 0U;
        for (; i_kpr1080 < 4U; i_kpr1080++) {
            uint32_t j_kpr1081 = i_kpr1080;
            vec_memcpy(elems_tile0 + (j_kpr1081 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1081 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1082 = 0U;
        for (; i_kpr1082 < 4U; i_kpr1082++) {
            uint32_t j_kpr1083 = i_kpr1082;
            vec_memcpy(col_ind_tile0 + (j_kpr1083 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1083 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1084 = 0U;
        for (; i_kpr1084 < to_; i_kpr1084++)
            elems_tile0[i_kpr1084 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1086 = 0U;
            for (; i_kpr1086 < 4U; i_kpr1086++) {
                uint32_t j_kpr1087 = i_kpr1086;
                vec_memcpy(elems_tile0 + (j_kpr1087 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1087 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1088 = 0U;
            for (; i_kpr1088 < 4U; i_kpr1088++) {
                uint32_t j_kpr1089 = i_kpr1088;
                vec_memcpy(col_ind_tile0 + (j_kpr1089 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1089 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1091 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1090 = 0U;
    for (; i_kpr1090 < hi_kpr1091; i_kpr1090++) {
        uint32_t j_kpr1092 = i_kpr1090;
        elems_tile0[j_kpr1092 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1092 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1092 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1092 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1093 = 0U;
    for (; i_kpr1093 < 2U; i_kpr1093++) {
        uint32_t j_kpr1094 = i_kpr1093;
        uint32_t off = n_idx + j_kpr1094 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1094 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x256x32_on
*/
static void
__hoisted_g_spmm_f32_512x256x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1095 = 0U;
        for (; i_kpr1095 < 4U; i_kpr1095++) {
            uint32_t j_kpr1096 = i_kpr1095;
            vec_memcpy(elems_tile0 + (j_kpr1096 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1096 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1097 = 0U;
        for (; i_kpr1097 < 4U; i_kpr1097++) {
            uint32_t j_kpr1098 = i_kpr1097;
            vec_memcpy(col_ind_tile0 + (j_kpr1098 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1098 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1099 = 0U;
        for (; i_kpr1099 < to_; i_kpr1099++)
            elems_tile0[i_kpr1099 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1101 = 0U;
            for (; i_kpr1101 < 4U; i_kpr1101++) {
                uint32_t j_kpr1102 = i_kpr1101;
                vec_memcpy(elems_tile0 + (j_kpr1102 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1102 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1103 = 0U;
            for (; i_kpr1103 < 4U; i_kpr1103++) {
                uint32_t j_kpr1104 = i_kpr1103;
                vec_memcpy(col_ind_tile0 + (j_kpr1104 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1104 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1106 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1105 = 0U;
    for (; i_kpr1105 < hi_kpr1106; i_kpr1105++) {
        uint32_t j_kpr1107 = i_kpr1105;
        elems_tile0[j_kpr1107 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1107 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1107 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1107 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1108 = 0U;
    for (; i_kpr1108 < 2U; i_kpr1108++) {
        uint32_t j_kpr1109 = i_kpr1108;
        uint32_t off = n_idx + j_kpr1109 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1109 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x256x64
*/
static void
__hoisted_g_spmm_f32_512x256x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1110 = 0U;
        for (; i_kpr1110 < 2U; i_kpr1110++) {
            uint32_t j_kpr1111 = i_kpr1110;
            vec_memcpy(elems_tile0 + (j_kpr1111 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1111 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1112 = 0U;
        for (; i_kpr1112 < 2U; i_kpr1112++) {
            uint32_t j_kpr1113 = i_kpr1112;
            vec_memcpy(col_ind_tile0 + (j_kpr1113 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1113 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1114 = 0U;
        for (; i_kpr1114 < to_; i_kpr1114++)
            elems_tile0[i_kpr1114 * 64U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1116 = 0U;
            for (; i_kpr1116 < 2U; i_kpr1116++) {
                uint32_t j_kpr1117 = i_kpr1116;
                vec_memcpy(elems_tile0 + (j_kpr1117 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1117 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1118 = 0U;
            for (; i_kpr1118 < 2U; i_kpr1118++) {
                uint32_t j_kpr1119 = i_kpr1118;
                vec_memcpy(col_ind_tile0 + (j_kpr1119 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1119 * 64U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1121 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1120 = 0U;
    for (; i_kpr1120 < hi_kpr1121; i_kpr1120++) {
        uint32_t j_kpr1122 = i_kpr1120;
        elems_tile0[j_kpr1122 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1122 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1122 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1122 * 64U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1123 = 0U;
    for (; i_kpr1123 < 1U; i_kpr1123++) {
        uint32_t j_kpr1124 = i_kpr1123;
        uint32_t off = n_idx + j_kpr1124 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1124 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x256x64_on
*/
static void
__hoisted_g_spmm_f32_512x256x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1125 = 0U;
        for (; i_kpr1125 < 2U; i_kpr1125++) {
            uint32_t j_kpr1126 = i_kpr1125;
            vec_memcpy(elems_tile0 + (j_kpr1126 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1126 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1127 = 0U;
        for (; i_kpr1127 < 2U; i_kpr1127++) {
            uint32_t j_kpr1128 = i_kpr1127;
            vec_memcpy(col_ind_tile0 + (j_kpr1128 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1128 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1129 = 0U;
        for (; i_kpr1129 < to_; i_kpr1129++)
            elems_tile0[i_kpr1129 * 64U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1131 = 0U;
            for (; i_kpr1131 < 2U; i_kpr1131++) {
                uint32_t j_kpr1132 = i_kpr1131;
                vec_memcpy(elems_tile0 + (j_kpr1132 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1132 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1133 = 0U;
            for (; i_kpr1133 < 2U; i_kpr1133++) {
                uint32_t j_kpr1134 = i_kpr1133;
                vec_memcpy(col_ind_tile0 + (j_kpr1134 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1134 * 64U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1136 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1135 = 0U;
    for (; i_kpr1135 < hi_kpr1136; i_kpr1135++) {
        uint32_t j_kpr1137 = i_kpr1135;
        elems_tile0[j_kpr1137 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1137 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1137 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1137 * 64U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1138 = 0U;
    for (; i_kpr1138 < 1U; i_kpr1138++) {
        uint32_t j_kpr1139 = i_kpr1138;
        uint32_t off = n_idx + j_kpr1139 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1139 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x512x16
*/
static void
__hoisted_g_spmm_f32_512x512x16_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1140 = 0U;
        for (; i_kpr1140 < 8U; i_kpr1140++) {
            uint32_t j_kpr1141 = i_kpr1140;
            vec_memcpy(elems_tile0 + (j_kpr1141 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1141 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1142 = 0U;
        for (; i_kpr1142 < 8U; i_kpr1142++) {
            uint32_t j_kpr1143 = i_kpr1142;
            vec_memcpy(col_ind_tile0 + (j_kpr1143 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1143 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1144 = 0U;
        for (; i_kpr1144 < to_; i_kpr1144++)
            elems_tile0[i_kpr1144 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1146 = 0U;
            for (; i_kpr1146 < 8U; i_kpr1146++) {
                uint32_t j_kpr1147 = i_kpr1146;
                vec_memcpy(elems_tile0 + (j_kpr1147 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1147 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1148 = 0U;
            for (; i_kpr1148 < 8U; i_kpr1148++) {
                uint32_t j_kpr1149 = i_kpr1148;
                vec_memcpy(col_ind_tile0 + (j_kpr1149 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1149 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1151 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1150 = 0U;
    for (; i_kpr1150 < hi_kpr1151; i_kpr1150++) {
        uint32_t j_kpr1152 = i_kpr1150;
        elems_tile0[j_kpr1152 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1152 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1152 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1152 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1153 = 0U;
    for (; i_kpr1153 < 8U; i_kpr1153++) {
        uint32_t j_kpr1154 = i_kpr1153;
        uint32_t off = n_idx + j_kpr1154 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1154 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_spmm_f32_512x512x16_on
*/
static void
__hoisted_g_spmm_f32_512x512x16_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1155 = 0U;
        for (; i_kpr1155 < 8U; i_kpr1155++) {
            uint32_t j_kpr1156 = i_kpr1155;
            vec_memcpy(elems_tile0 + (j_kpr1156 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1156 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1157 = 0U;
        for (; i_kpr1157 < 8U; i_kpr1157++) {
            uint32_t j_kpr1158 = i_kpr1157;
            vec_memcpy(col_ind_tile0 + (j_kpr1158 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1158 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1159 = 0U;
        for (; i_kpr1159 < to_; i_kpr1159++)
            elems_tile0[i_kpr1159 * 16U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1161 = 0U;
            for (; i_kpr1161 < 8U; i_kpr1161++) {
                uint32_t j_kpr1162 = i_kpr1161;
                vec_memcpy(elems_tile0 + (j_kpr1162 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1162 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1163 = 0U;
            for (; i_kpr1163 < 8U; i_kpr1163++) {
                uint32_t j_kpr1164 = i_kpr1163;
                vec_memcpy(col_ind_tile0 + (j_kpr1164 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1164 * 16U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 16U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1166 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1165 = 0U;
    for (; i_kpr1165 < hi_kpr1166; i_kpr1165++) {
        uint32_t j_kpr1167 = i_kpr1165;
        elems_tile0[j_kpr1167 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1167 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1167 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1167 * 16U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1168 = 0U;
    for (; i_kpr1168 < 8U; i_kpr1168++) {
        uint32_t j_kpr1169 = i_kpr1168;
        uint32_t off = n_idx + j_kpr1169 * 16U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1169 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x512x32
*/
static void
__hoisted_g_spmm_f32_512x512x32_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1170 = 0U;
        for (; i_kpr1170 < 4U; i_kpr1170++) {
            uint32_t j_kpr1171 = i_kpr1170;
            vec_memcpy(elems_tile0 + (j_kpr1171 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1171 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1172 = 0U;
        for (; i_kpr1172 < 4U; i_kpr1172++) {
            uint32_t j_kpr1173 = i_kpr1172;
            vec_memcpy(col_ind_tile0 + (j_kpr1173 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1173 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1174 = 0U;
        for (; i_kpr1174 < to_; i_kpr1174++)
            elems_tile0[i_kpr1174 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1176 = 0U;
            for (; i_kpr1176 < 4U; i_kpr1176++) {
                uint32_t j_kpr1177 = i_kpr1176;
                vec_memcpy(elems_tile0 + (j_kpr1177 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1177 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1178 = 0U;
            for (; i_kpr1178 < 4U; i_kpr1178++) {
                uint32_t j_kpr1179 = i_kpr1178;
                vec_memcpy(col_ind_tile0 + (j_kpr1179 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1179 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1181 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1180 = 0U;
    for (; i_kpr1180 < hi_kpr1181; i_kpr1180++) {
        uint32_t j_kpr1182 = i_kpr1180;
        elems_tile0[j_kpr1182 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1182 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1182 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1182 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1183 = 0U;
    for (; i_kpr1183 < 4U; i_kpr1183++) {
        uint32_t j_kpr1184 = i_kpr1183;
        uint32_t off = n_idx + j_kpr1184 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1184 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_spmm_f32_512x512x32_on
*/
static void
__hoisted_g_spmm_f32_512x512x32_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1185 = 0U;
        for (; i_kpr1185 < 4U; i_kpr1185++) {
            uint32_t j_kpr1186 = i_kpr1185;
            vec_memcpy(elems_tile0 + (j_kpr1186 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1186 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1187 = 0U;
        for (; i_kpr1187 < 4U; i_kpr1187++) {
            uint32_t j_kpr1188 = i_kpr1187;
            vec_memcpy(col_ind_tile0 + (j_kpr1188 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1188 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1189 = 0U;
        for (; i_kpr1189 < to_; i_kpr1189++)
            elems_tile0[i_kpr1189 * 32U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1191 = 0U;
            for (; i_kpr1191 < 4U; i_kpr1191++) {
                uint32_t j_kpr1192 = i_kpr1191;
                vec_memcpy(elems_tile0 + (j_kpr1192 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1192 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1193 = 0U;
            for (; i_kpr1193 < 4U; i_kpr1193++) {
                uint32_t j_kpr1194 = i_kpr1193;
                vec_memcpy(col_ind_tile0 + (j_kpr1194 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1194 * 32U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 32U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1196 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1195 = 0U;
    for (; i_kpr1195 < hi_kpr1196; i_kpr1195++) {
        uint32_t j_kpr1197 = i_kpr1195;
        elems_tile0[j_kpr1197 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1197 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1197 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1197 * 32U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1198 = 0U;
    for (; i_kpr1198 < 4U; i_kpr1198++) {
        uint32_t j_kpr1199 = i_kpr1198;
        uint32_t off = n_idx + j_kpr1199 * 32U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1199 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x512x64
*/
static void
__hoisted_g_spmm_f32_512x512x64_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1200 = 0U;
        for (; i_kpr1200 < 2U; i_kpr1200++) {
            uint32_t j_kpr1201 = i_kpr1200;
            vec_memcpy(elems_tile0 + (j_kpr1201 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1201 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1202 = 0U;
        for (; i_kpr1202 < 2U; i_kpr1202++) {
            uint32_t j_kpr1203 = i_kpr1202;
            vec_memcpy(col_ind_tile0 + (j_kpr1203 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1203 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1204 = 0U;
        for (; i_kpr1204 < to_; i_kpr1204++)
            elems_tile0[i_kpr1204 * 64U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1206 = 0U;
            for (; i_kpr1206 < 2U; i_kpr1206++) {
                uint32_t j_kpr1207 = i_kpr1206;
                vec_memcpy(elems_tile0 + (j_kpr1207 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1207 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1208 = 0U;
            for (; i_kpr1208 < 2U; i_kpr1208++) {
                uint32_t j_kpr1209 = i_kpr1208;
                vec_memcpy(col_ind_tile0 + (j_kpr1209 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1209 * 64U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1211 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1210 = 0U;
    for (; i_kpr1210 < hi_kpr1211; i_kpr1210++) {
        uint32_t j_kpr1212 = i_kpr1210;
        elems_tile0[j_kpr1212 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1212 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1212 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1212 * 64U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1213 = 0U;
    for (; i_kpr1213 < 2U; i_kpr1213++) {
        uint32_t j_kpr1214 = i_kpr1213;
        uint32_t off = n_idx + j_kpr1214 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1214 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_spmm_f32_512x512x64_on
*/
static void
__hoisted_g_spmm_f32_512x512x64_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1215 = 0U;
        for (; i_kpr1215 < 2U; i_kpr1215++) {
            uint32_t j_kpr1216 = i_kpr1215;
            vec_memcpy(elems_tile0 + (j_kpr1216 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1216 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1217 = 0U;
        for (; i_kpr1217 < 2U; i_kpr1217++) {
            uint32_t j_kpr1218 = i_kpr1217;
            vec_memcpy(col_ind_tile0 + (j_kpr1218 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1218 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1219 = 0U;
        for (; i_kpr1219 < to_; i_kpr1219++)
            elems_tile0[i_kpr1219 * 64U + threadIdx.x] = 0.0f;
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
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1221 = 0U;
            for (; i_kpr1221 < 2U; i_kpr1221++) {
                uint32_t j_kpr1222 = i_kpr1221;
                vec_memcpy(elems_tile0 + (j_kpr1222 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1222 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1223 = 0U;
            for (; i_kpr1223 < 2U; i_kpr1223++) {
                uint32_t j_kpr1224 = i_kpr1223;
                vec_memcpy(col_ind_tile0 + (j_kpr1224 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1224 * 64U + threadIdx.x) * 4U));
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
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 64U * 4U)));
                            uint32_t ix = 0U;
                            for (; ix < 4U; ix++)
                                out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1226 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1225 = 0U;
    for (; i_kpr1225 < hi_kpr1226; i_kpr1225++) {
        uint32_t j_kpr1227 = i_kpr1225;
        elems_tile0[j_kpr1227 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1227 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1227 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1227 * 64U + threadIdx.x];
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1228 = 0U;
    for (; i_kpr1228 < 2U; i_kpr1228++) {
        uint32_t j_kpr1229 = i_kpr1228;
        uint32_t off = n_idx + j_kpr1229 * 64U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1229 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_spmm_f32_512x512x128
*/
static void
__hoisted_g_spmm_f32_512x512x128_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1230 = 0U;
        for (; i_kpr1230 < 1U; i_kpr1230++) {
            uint32_t j_kpr1231 = i_kpr1230;
            vec_memcpy(elems_tile0 + (j_kpr1231 * 128U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1231 * 128U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1232 = 0U;
        for (; i_kpr1232 < 1U; i_kpr1232++) {
            uint32_t j_kpr1233 = i_kpr1232;
            vec_memcpy(col_ind_tile0 + (j_kpr1233 * 128U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1233 * 128U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i_kpr1234 = 0U;
        for (; i_kpr1234 < to_; i_kpr1234++)
            elems_tile0[i_kpr1234 * 128U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 128U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1236 = 0U;
            for (; i_kpr1236 < 1U; i_kpr1236++) {
                uint32_t j_kpr1237 = i_kpr1236;
                vec_memcpy(elems_tile0 + (j_kpr1237 * 128U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1237 * 128U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1238 = 0U;
            for (; i_kpr1238 < 1U; i_kpr1238++) {
                uint32_t j_kpr1239 = i_kpr1238;
                vec_memcpy(
                    col_ind_tile0 + (j_kpr1239 * 128U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1239 * 128U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 128U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1241 = (__anf01 + 127U - threadIdx.x) / 128U;
    uint32_t i_kpr1240 = 0U;
    for (; i_kpr1240 < hi_kpr1241; i_kpr1240++) {
        uint32_t j_kpr1242 = i_kpr1240;
        elems_tile0[j_kpr1242 * 128U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1242 * 128U + threadIdx.x];
        col_ind_tile0[j_kpr1242 * 128U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1242 * 128U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 128U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1243 = 0U;
    for (; i_kpr1243 < 1U; i_kpr1243++) {
        uint32_t j_kpr1244 = i_kpr1243;
        uint32_t off = n_idx + j_kpr1244 * 128U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1244 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_spmm_f32_512x512x128_on
*/
static void
__hoisted_g_spmm_f32_512x512x128_on_0(uint32_t *row_indices, uint32_t rows,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t cols, float *gB,
    float *gC)
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
        uint32_t i_kpr1245 = 0U;
        for (; i_kpr1245 < 1U; i_kpr1245++) {
            uint32_t j_kpr1246 = i_kpr1245;
            vec_memcpy(elems_tile0 + (j_kpr1246 * 128U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1246 * 128U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1247 = 0U;
        for (; i_kpr1247 < 1U; i_kpr1247++) {
            uint32_t j_kpr1248 = i_kpr1247;
            vec_memcpy(col_ind_tile0 + (j_kpr1248 * 128U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1248 * 128U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i_kpr1249 = 0U;
        for (; i_kpr1249 < to_; i_kpr1249++)
            elems_tile0[i_kpr1249 * 128U + threadIdx.x] = 0.0f;
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
                    vec_memcpy(lchunk,
                        gB + (cols * kr + (n_idx + __anf01 * 128U * 4U)));
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
        idx = 1U;
        nnz -= 512U;
        for (; nnz >= 512U; nnz -= 512U) {
            uint32_t off = ri_ + idx * 512U;
            __syncthreads();
            uint32_t i_kpr1251 = 0U;
            for (; i_kpr1251 < 1U; i_kpr1251++) {
                uint32_t j_kpr1252 = i_kpr1251;
                vec_memcpy(elems_tile0 + (j_kpr1252 * 128U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1252 * 128U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1253 = 0U;
            for (; i_kpr1253 < 1U; i_kpr1253++) {
                uint32_t j_kpr1254 = i_kpr1253;
                vec_memcpy(
                    col_ind_tile0 + (j_kpr1254 * 128U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1254 * 128U + threadIdx.x) * 4U));
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
                            gB + (cols * kr + (n_idx + __anf03 * 128U * 4U)));
                        uint32_t ix = 0U;
                        for (; ix < 4U; ix++)
                            out[__anf1 * 4U + ix] += kx * lchunk[ix];
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
    uint32_t hi_kpr1256 = (__anf01 + 127U - threadIdx.x) / 128U;
    uint32_t i_kpr1255 = 0U;
    for (; i_kpr1255 < hi_kpr1256; i_kpr1255++) {
        uint32_t j_kpr1257 = i_kpr1255;
        elems_tile0[j_kpr1257 * 128U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1257 * 128U + threadIdx.x];
        col_ind_tile0[j_kpr1257 * 128U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1257 * 128U + threadIdx.x];
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
                    lchunk, gB + (cols * kr + (n_idx + __anf02 * 128U * 4U)));
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1258 = 0U;
    for (; i_kpr1258 < 1U; i_kpr1258++) {
        uint32_t j_kpr1259 = i_kpr1258;
        uint32_t off = n_idx + j_kpr1259 * 128U * 4U;
        if (off < cols) {
            uint32_t address = cols * m_idx + off;
            float combined[4U];
            memset(combined, 0U, 4U * sizeof(float));
            vec_memcpy(combined, gC + address);
            uint32_t lane = 0U;
            for (; lane < 4U; lane++)
                combined[lane] = out[j_kpr1259 * 4U + lane];
            vec_memcpy(gC + address, combined);
        }
    }
}

void Klas_SPMM_spmm_u32(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix__uint32_t gA, uint32_t *row_indices,
    uint32_t *gB, uint32_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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

void Klas_SPMM_g_spmm_f16_f16_f32_f32_32x32x4(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, half *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_f16_f32_f32_32x32x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_f16_f32_f32_32x32x4_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f16_f16_f32_f32_32x32x4_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, half *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_f16_f32_f32_32x32x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_f16_f32_f32_32x32x4_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_bf16_bf16_f32_f32_32x32x4(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix____nv_bfloat16 gA,
    uint32_t *row_indices, __nv_bfloat16 *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 32U +
                   (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 32U +
            (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_bf16_bf16_f32_f32_32x32x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_bf16_bf16_f32_f32_32x32x4_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(__nv_bfloat16) * 32U +
            (uint32_t) sizeof(uint32_t) * 32U,
        s, row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_bf16_bf16_f32_f32_32x32x4_on(uint32_t rows,
    uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix____nv_bfloat16 gA, uint32_t *row_indices,
    __nv_bfloat16 *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 32U +
                   (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 32U +
            (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(
            __hoisted_g_spmm_bf16_bf16_f32_f32_32x32x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 32U +
                (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_bf16_bf16_f32_f32_32x32x4_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(__nv_bfloat16) * 32U +
            (uint32_t) sizeof(uint32_t) * 32U,
        s, row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f16_f32_f16_f32_32x32x4(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, float *gB, half *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_f32_f16_f32_32x32x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_f32_f16_f32_32x32x4_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f16_f32_f16_f32_32x32x4_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, float *gB, half *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_f32_f16_f32_32x32x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_f32_f16_f32_32x32x4_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f16_bf16_f32_f64_32x32x4(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, __nv_bfloat16 *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_bf16_f32_f64_32x32x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_bf16_f32_f64_32x32x4_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f16_bf16_f32_f64_32x32x4_on(uint32_t rows,
    uint32_t shared, uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, __nv_bfloat16 *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(
            __hoisted_g_spmm_f16_bf16_f32_f64_32x32x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_bf16_f32_f64_32x32x4_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_f16_f32_f32_f64_32x32x4(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_f32_f32_f64_32x32x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_f32_f32_f64_32x32x4_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_f16_f32_f32_f64_32x32x4_on(uint32_t rows, uint32_t shared,
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__half gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_spmm_f16_f32_f32_f64_32x32x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_f16_f32_f32_f64_32x32x4_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC);
}

void Klas_SPMM_g_spmm_mapped_f16_f32_f16_f32_32x32x4(float alpha, float beta,
    uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t *row_indices, float *gB,
    half *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(
            __hoisted_g_spmm_mapped_f16_f32_f16_f32_32x32x4_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_mapped_f16_f32_f16_f32_32x32x4_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_SPMM_g_spmm_mapped_f16_f32_f16_f32_32x32x4_on(float alpha, float beta,
    uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix__half gA, uint32_t *row_indices, float *gB,
    half *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U);
    if ((uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U >=
        49152U)
        MUST(cudaFuncSetAttribute(
            __hoisted_g_spmm_mapped_f16_f32_f16_f32_32x32x4_on_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U));
    KPR_KCALL(__hoisted_g_spmm_mapped_f16_f32_f16_f32_32x32x4_on_0,
        rows * (cols / 32U + (uint32_t) (cols % 32U != 0U)), 4U,
        (uint32_t) sizeof(half) * 32U + (uint32_t) sizeof(uint32_t) * 32U, s,
        row_indices, rows, gA, cols, gB, gC, beta, alpha);
}

void Klas_SPMM_g_spmm_f32_32x4x1(uint32_t rows, uint32_t shared, uint32_t cols,
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
    float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    cudaStream_t s = KPR_FRESH_STREAM();
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
    uint32_t cols, Kuiper_Sparse_Matrix_csr_matrix__float gA,
    uint32_t *row_indices, float *gB, float *gC, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(shared);
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
    Kuiper_Sparse_Matrix_csr_matrix__float gA, uint32_t *row_indices, float *gB,
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
