
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr15 = 0U;
    for (; i_kpr15 < 1U; i_kpr15++) {
        uint32_t j_kpr16 = i_kpr15;
        uint32_t off = n_idx + j_kpr16 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr13 = 0U;
            for (; i_kpr13 < 1U; i_kpr13++) {
                uint32_t j_kpr14 = i_kpr13;
                uint32_t start = j_kpr16 * 4U + j_kpr14 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr14 * 4U);
                uint32_t combined[4U] = {0U};
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr17 = 0U;
        for (; i_kpr17 < 2U; i_kpr17++) {
            uint32_t j_kpr18 = i_kpr17;
            vec_memcpy(elems_tile0 + (j_kpr18 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr18 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr19 = 0U;
        for (; i_kpr19 < 2U; i_kpr19++) {
            uint32_t j_kpr20 = i_kpr19;
            vec_memcpy(col_ind_tile0 + (j_kpr20 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr20 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr21 = 0U;
        for (; i_kpr21 < to_; i_kpr21++)
            elems_tile0[i_kpr21 * 64U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr23 = 0U;
            for (; i_kpr23 < 2U; i_kpr23++) {
                uint32_t j_kpr24 = i_kpr23;
                vec_memcpy(elems_tile0 + (j_kpr24 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr24 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr25 = 0U;
            for (; i_kpr25 < 2U; i_kpr25++) {
                uint32_t j_kpr26 = i_kpr25;
                vec_memcpy(col_ind_tile0 + (j_kpr26 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr26 * 64U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 64U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr28 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr27 = 0U;
    for (; i_kpr27 < hi_kpr28; i_kpr27++) {
        uint32_t j_kpr29 = i_kpr27;
        elems_tile0[j_kpr29 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr29 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr29 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr29 * 64U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr32 = 0U;
    for (; i_kpr32 < 2U; i_kpr32++) {
        uint32_t j_kpr33 = i_kpr32;
        uint32_t off = n_idx + j_kpr33 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr30 = 0U;
            for (; i_kpr30 < 1U; i_kpr30++) {
                uint32_t j_kpr31 = i_kpr30;
                uint32_t start = j_kpr33 * 4U + j_kpr31 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr31 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr34 = 0U;
        for (; i_kpr34 < 1U; i_kpr34++) {
            uint32_t j_kpr35 = i_kpr34;
            vec_memcpy(elems_tile0 + (j_kpr35 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr35 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr36 = 0U;
        for (; i_kpr36 < 2U; i_kpr36++) {
            uint32_t j_kpr37 = i_kpr36;
            vec_memcpy(col_ind_tile0 + (j_kpr37 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr37 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr38 = 0U;
        for (; i_kpr38 < to_; i_kpr38++) {
            uint32_t j_kpr39 = i_kpr38;
            elems_tile0[j_kpr39 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    half lchunk[8U];
                    for (uint32_t _i = 0U; _i < 8U; ++_i)
                        lchunk[_i] = __float2half_rn(0.0f);
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 8U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 8U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        half __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + __half2float(kx) * __half2float(__anf11);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr40 = 0U;
            for (; i_kpr40 < 1U; i_kpr40++) {
                uint32_t j_kpr41 = i_kpr40;
                vec_memcpy(elems_tile0 + (j_kpr41 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr41 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr42 = 0U;
            for (; i_kpr42 < 2U; i_kpr42++) {
                uint32_t j_kpr43 = i_kpr42;
                vec_memcpy(col_ind_tile0 + (j_kpr43 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr43 * 4U + threadIdx.x) * 4U));
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
                        half lchunk[8U];
                        for (uint32_t _i = 0U; _i < 8U; ++_i)
                            lchunk[_i] = __float2half_rn(0.0f);
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 8U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 8U));
                            k2 = kv1 + 1U;
                        }
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            half __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 +
                                __half2float(kx) * __half2float(__anf11);
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
    uint32_t hi_kpr45 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr44 = 0U;
    for (; i_kpr44 < hi_kpr45; i_kpr44++) {
        uint32_t j_kpr46 = i_kpr44;
        elems_tile0[j_kpr46 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr46 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr46 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr46 * 4U + threadIdx.x];
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
                half lchunk[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    lchunk[_i] = __float2half_rn(0.0f);
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 8U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 8U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    half __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + __half2float(kx) * __half2float(__anf11);
                }
            }
        }
    }
    uint32_t i_kpr49 = 0U;
    for (; i_kpr49 < 1U; i_kpr49++) {
        uint32_t j_kpr50 = i_kpr49;
        uint32_t off = n_idx + j_kpr50 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr47 = 0U;
            for (; i_kpr47 < 2U; i_kpr47++) {
                uint32_t j_kpr48 = i_kpr47;
                uint32_t start = j_kpr50 * 8U + j_kpr48 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr48 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr51 = 0U;
        for (; i_kpr51 < 1U; i_kpr51++) {
            uint32_t j_kpr52 = i_kpr51;
            vec_memcpy(elems_tile0 + (j_kpr52 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr52 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr53 = 0U;
        for (; i_kpr53 < 2U; i_kpr53++) {
            uint32_t j_kpr54 = i_kpr53;
            vec_memcpy(col_ind_tile0 + (j_kpr54 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr54 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr55 = 0U;
        for (; i_kpr55 < to_; i_kpr55++) {
            uint32_t j_kpr56 = i_kpr55;
            elems_tile0[j_kpr56 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    half lchunk[8U];
                    for (uint32_t _i = 0U; _i < 8U; ++_i)
                        lchunk[_i] = __float2half_rn(0.0f);
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 8U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 8U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        half __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + __half2float(kx) * __half2float(__anf11);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr57 = 0U;
            for (; i_kpr57 < 1U; i_kpr57++) {
                uint32_t j_kpr58 = i_kpr57;
                vec_memcpy(elems_tile0 + (j_kpr58 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr58 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr59 = 0U;
            for (; i_kpr59 < 2U; i_kpr59++) {
                uint32_t j_kpr60 = i_kpr59;
                vec_memcpy(col_ind_tile0 + (j_kpr60 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr60 * 4U + threadIdx.x) * 4U));
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
                        half lchunk[8U];
                        for (uint32_t _i = 0U; _i < 8U; ++_i)
                            lchunk[_i] = __float2half_rn(0.0f);
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 8U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 8U));
                            k2 = kv1 + 1U;
                        }
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            half __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 +
                                __half2float(kx) * __half2float(__anf11);
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
    uint32_t hi_kpr62 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr61 = 0U;
    for (; i_kpr61 < hi_kpr62; i_kpr61++) {
        uint32_t j_kpr63 = i_kpr61;
        elems_tile0[j_kpr63 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr63 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr63 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr63 * 4U + threadIdx.x];
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
                half lchunk[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    lchunk[_i] = __float2half_rn(0.0f);
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 8U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 8U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    half __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + __half2float(kx) * __half2float(__anf11);
                }
            }
        }
    }
    uint32_t i_kpr66 = 0U;
    for (; i_kpr66 < 1U; i_kpr66++) {
        uint32_t j_kpr67 = i_kpr66;
        uint32_t off = n_idx + j_kpr67 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr64 = 0U;
            for (; i_kpr64 < 2U; i_kpr64++) {
                uint32_t j_kpr65 = i_kpr64;
                uint32_t start = j_kpr67 * 8U + j_kpr65 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr65 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
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
        uint32_t i_kpr68 = 0U;
        for (; i_kpr68 < 1U; i_kpr68++) {
            uint32_t j_kpr69 = i_kpr68;
            vec_memcpy(elems_tile0 + (j_kpr69 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr69 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr70 = 0U;
        for (; i_kpr70 < 2U; i_kpr70++) {
            uint32_t j_kpr71 = i_kpr70;
            vec_memcpy(col_ind_tile0 + (j_kpr71 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr71 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr72 = 0U;
        for (; i_kpr72 < to_; i_kpr72++) {
            uint32_t j_kpr73 = i_kpr72;
            elems_tile0[j_kpr73 * 4U + threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                __nv_bfloat16 kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    __nv_bfloat16 lchunk[8U];
                    for (uint32_t _i = 0U; _i < 8U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 8U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 8U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 +
                            __bfloat162float(kx) * __bfloat162float(__anf11);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr74 = 0U;
            for (; i_kpr74 < 1U; i_kpr74++) {
                uint32_t j_kpr75 = i_kpr74;
                vec_memcpy(elems_tile0 + (j_kpr75 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr75 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr76 = 0U;
            for (; i_kpr76 < 2U; i_kpr76++) {
                uint32_t j_kpr77 = i_kpr76;
                vec_memcpy(col_ind_tile0 + (j_kpr77 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr77 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    __nv_bfloat16 kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        __nv_bfloat16 lchunk[8U];
                        for (uint32_t _i = 0U; _i < 8U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 8U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 8U));
                            k2 = kv1 + 1U;
                        }
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + __bfloat162float(kx) *
                                             __bfloat162float(__anf11);
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
    uint32_t hi_kpr79 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr78 = 0U;
    for (; i_kpr78 < hi_kpr79; i_kpr78++) {
        uint32_t j_kpr80 = i_kpr78;
        elems_tile0[j_kpr80 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr80 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr80 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr80 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            __nv_bfloat16 kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                __nv_bfloat16 lchunk[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    lchunk[_i] = __float2bfloat16(0.0f);
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 8U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 8U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    __nv_bfloat16 __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 +
                        __bfloat162float(kx) * __bfloat162float(__anf11);
                }
            }
        }
    }
    uint32_t i_kpr83 = 0U;
    for (; i_kpr83 < 1U; i_kpr83++) {
        uint32_t j_kpr84 = i_kpr83;
        uint32_t off = n_idx + j_kpr84 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr81 = 0U;
            for (; i_kpr81 < 2U; i_kpr81++) {
                uint32_t j_kpr82 = i_kpr81;
                uint32_t start = j_kpr84 * 8U + j_kpr82 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr82 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
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
        uint32_t i_kpr85 = 0U;
        for (; i_kpr85 < 1U; i_kpr85++) {
            uint32_t j_kpr86 = i_kpr85;
            vec_memcpy(elems_tile0 + (j_kpr86 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr86 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr87 = 0U;
        for (; i_kpr87 < 2U; i_kpr87++) {
            uint32_t j_kpr88 = i_kpr87;
            vec_memcpy(col_ind_tile0 + (j_kpr88 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr88 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr89 = 0U;
        for (; i_kpr89 < to_; i_kpr89++) {
            uint32_t j_kpr90 = i_kpr89;
            elems_tile0[j_kpr90 * 4U + threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        if (n_idx < cols) {
            uint32_t k = 0U;
            for (; k < 32U; k++) {
                uint32_t kv = k;
                uint32_t kr = col_ind_tile0[kv];
                __nv_bfloat16 kx = elems_tile0[kv];
                uint32_t k1 = 0U;
                for (; k1 < 1U; k1++) {
                    uint32_t __anf1 = k1;
                    uint32_t __anf01 = k1;
                    __nv_bfloat16 lchunk[8U];
                    for (uint32_t _i = 0U; _i < 8U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 8U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 8U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        float __anf3 = out[__anf1 * 8U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 +
                            __bfloat162float(kx) * __bfloat162float(__anf11);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr91 = 0U;
            for (; i_kpr91 < 1U; i_kpr91++) {
                uint32_t j_kpr92 = i_kpr91;
                vec_memcpy(elems_tile0 + (j_kpr92 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr92 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr93 = 0U;
            for (; i_kpr93 < 2U; i_kpr93++) {
                uint32_t j_kpr94 = i_kpr93;
                vec_memcpy(col_ind_tile0 + (j_kpr94 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr94 * 4U + threadIdx.x) * 4U));
            }
            __syncthreads();
            if (n_idx < cols) {
                uint32_t k = 0U;
                for (; k < 32U; k++) {
                    uint32_t kv = k;
                    uint32_t kr = col_ind_tile0[kv];
                    __nv_bfloat16 kx = elems_tile0[kv];
                    uint32_t k1 = 0U;
                    for (; k1 < 1U; k1++) {
                        uint32_t __anf1 = k1;
                        uint32_t __anf03 = k1;
                        __nv_bfloat16 lchunk[8U];
                        for (uint32_t _i = 0U; _i < 8U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 8U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 8U));
                            k2 = kv1 + 1U;
                        }
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            float __anf3 = out[__anf1 * 8U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + __bfloat162float(kx) *
                                             __bfloat162float(__anf11);
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
    uint32_t hi_kpr96 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr95 = 0U;
    for (; i_kpr95 < hi_kpr96; i_kpr95++) {
        uint32_t j_kpr97 = i_kpr95;
        elems_tile0[j_kpr97 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr97 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr97 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr97 * 4U + threadIdx.x];
    }
    __syncthreads();
    if (n_idx < cols) {
        uint32_t k = 0U;
        for (; k < __anf01; k++) {
            uint32_t kv = k;
            uint32_t kr = col_ind_tile0[kv];
            __nv_bfloat16 kx = elems_tile0[kv];
            uint32_t k1 = 0U;
            for (; k1 < 1U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                __nv_bfloat16 lchunk[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    lchunk[_i] = __float2bfloat16(0.0f);
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 8U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 8U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    float __anf3 = out[__anf1 * 8U + ix];
                    __nv_bfloat16 __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 +
                        __bfloat162float(kx) * __bfloat162float(__anf11);
                }
            }
        }
    }
    uint32_t i_kpr100 = 0U;
    for (; i_kpr100 < 1U; i_kpr100++) {
        uint32_t j_kpr101 = i_kpr100;
        uint32_t off = n_idx + j_kpr101 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr98 = 0U;
            for (; i_kpr98 < 2U; i_kpr98++) {
                uint32_t j_kpr99 = i_kpr98;
                uint32_t start = j_kpr101 * 8U + j_kpr99 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr99 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr102 = 0U;
        for (; i_kpr102 < 1U; i_kpr102++) {
            uint32_t j_kpr103 = i_kpr102;
            vec_memcpy(elems_tile0 + (j_kpr103 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr103 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr104 = 0U;
        for (; i_kpr104 < 2U; i_kpr104++) {
            uint32_t j_kpr105 = i_kpr104;
            vec_memcpy(col_ind_tile0 + (j_kpr105 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr105 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr106 = 0U;
        for (; i_kpr106 < to_; i_kpr106++) {
            uint32_t j_kpr107 = i_kpr106;
            elems_tile0[j_kpr107 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    uint32_t k2 = 0U;
                    while (k2 < 2U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr108 = 0U;
            for (; i_kpr108 < 1U; i_kpr108++) {
                uint32_t j_kpr109 = i_kpr108;
                vec_memcpy(elems_tile0 + (j_kpr109 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr109 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr110 = 0U;
            for (; i_kpr110 < 2U; i_kpr110++) {
                uint32_t j_kpr111 = i_kpr110;
                vec_memcpy(col_ind_tile0 + (j_kpr111 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr111 * 4U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 2U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr113 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr112 = 0U;
    for (; i_kpr112 < hi_kpr113; i_kpr112++) {
        uint32_t j_kpr114 = i_kpr112;
        elems_tile0[j_kpr114 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr114 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr114 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr114 * 4U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 2U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
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
    uint32_t i_kpr117 = 0U;
    for (; i_kpr117 < 1U; i_kpr117++) {
        uint32_t j_kpr118 = i_kpr117;
        uint32_t off = n_idx + j_kpr118 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr115 = 0U;
            for (; i_kpr115 < 1U; i_kpr115++) {
                uint32_t j_kpr116 = i_kpr115;
                uint32_t start = j_kpr118 * 8U + j_kpr116 * 8U;
                uint32_t address = cols * m_idx + (off + j_kpr116 * 8U);
                half combined[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    combined[_i] = __float2half_rn(0.0f);
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 8U; lane++) {
                    uint32_t __anf4 = lane;
                    combined[__anf4] = __float2half_rn(out[start + lane]);
                }
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr119 = 0U;
        for (; i_kpr119 < 1U; i_kpr119++) {
            uint32_t j_kpr120 = i_kpr119;
            vec_memcpy(elems_tile0 + (j_kpr120 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr120 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr121 = 0U;
        for (; i_kpr121 < 2U; i_kpr121++) {
            uint32_t j_kpr122 = i_kpr121;
            vec_memcpy(col_ind_tile0 + (j_kpr122 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr122 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr123 = 0U;
        for (; i_kpr123 < to_; i_kpr123++) {
            uint32_t j_kpr124 = i_kpr123;
            elems_tile0[j_kpr124 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    uint32_t k2 = 0U;
                    while (k2 < 2U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr125 = 0U;
            for (; i_kpr125 < 1U; i_kpr125++) {
                uint32_t j_kpr126 = i_kpr125;
                vec_memcpy(elems_tile0 + (j_kpr126 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr126 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr127 = 0U;
            for (; i_kpr127 < 2U; i_kpr127++) {
                uint32_t j_kpr128 = i_kpr127;
                vec_memcpy(col_ind_tile0 + (j_kpr128 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr128 * 4U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 2U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr130 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr129 = 0U;
    for (; i_kpr129 < hi_kpr130; i_kpr129++) {
        uint32_t j_kpr131 = i_kpr129;
        elems_tile0[j_kpr131 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr131 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr131 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr131 * 4U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 2U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
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
    uint32_t i_kpr134 = 0U;
    for (; i_kpr134 < 1U; i_kpr134++) {
        uint32_t j_kpr135 = i_kpr134;
        uint32_t off = n_idx + j_kpr135 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr132 = 0U;
            for (; i_kpr132 < 1U; i_kpr132++) {
                uint32_t j_kpr133 = i_kpr132;
                uint32_t start = j_kpr135 * 8U + j_kpr133 * 8U;
                uint32_t address = cols * m_idx + (off + j_kpr133 * 8U);
                half combined[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    combined[_i] = __float2half_rn(0.0f);
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 8U; lane++) {
                    uint32_t __anf4 = lane;
                    combined[__anf4] = __float2half_rn(out[start + lane]);
                }
                vec_memcpy(gC + address, combined);
            }
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
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
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
        uint32_t i_kpr136 = 0U;
        for (; i_kpr136 < 1U; i_kpr136++) {
            uint32_t j_kpr137 = i_kpr136;
            vec_memcpy(elems_tile0 + (j_kpr137 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr137 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr138 = 0U;
        for (; i_kpr138 < 2U; i_kpr138++) {
            uint32_t j_kpr139 = i_kpr138;
            vec_memcpy(col_ind_tile0 + (j_kpr139 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr139 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr140 = 0U;
        for (; i_kpr140 < to_; i_kpr140++) {
            uint32_t j_kpr141 = i_kpr140;
            elems_tile0[j_kpr141 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    __nv_bfloat16 lchunk[8U];
                    for (uint32_t _i = 0U; _i < 8U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 8U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 8U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        double __anf3 = out[__anf1 * 8U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + (double) __half2float(kx) *
                                         (double) __bfloat162float(__anf11);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr142 = 0U;
            for (; i_kpr142 < 1U; i_kpr142++) {
                uint32_t j_kpr143 = i_kpr142;
                vec_memcpy(elems_tile0 + (j_kpr143 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr143 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr144 = 0U;
            for (; i_kpr144 < 2U; i_kpr144++) {
                uint32_t j_kpr145 = i_kpr144;
                vec_memcpy(col_ind_tile0 + (j_kpr145 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr145 * 4U + threadIdx.x) * 4U));
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
                        __nv_bfloat16 lchunk[8U];
                        for (uint32_t _i = 0U; _i < 8U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 8U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 8U));
                            k2 = kv1 + 1U;
                        }
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            double __anf3 = out[__anf1 * 8U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + (double) __half2float(kx) *
                                             (double) __bfloat162float(__anf11);
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
    uint32_t hi_kpr147 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr146 = 0U;
    for (; i_kpr146 < hi_kpr147; i_kpr146++) {
        uint32_t j_kpr148 = i_kpr146;
        elems_tile0[j_kpr148 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr148 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr148 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr148 * 4U + threadIdx.x];
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
                __nv_bfloat16 lchunk[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    lchunk[_i] = __float2bfloat16(0.0f);
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 8U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 8U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    double __anf3 = out[__anf1 * 8U + ix];
                    __nv_bfloat16 __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + (double) __half2float(kx) *
                                     (double) __bfloat162float(__anf11);
                }
            }
        }
    }
    uint32_t i_kpr151 = 0U;
    for (; i_kpr151 < 1U; i_kpr151++) {
        uint32_t j_kpr152 = i_kpr151;
        uint32_t off = n_idx + j_kpr152 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr149 = 0U;
            for (; i_kpr149 < 2U; i_kpr149++) {
                uint32_t j_kpr150 = i_kpr149;
                uint32_t start = j_kpr152 * 8U + j_kpr150 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr150 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = (float) out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
    uint32_t n_idx = blockIdx.x / rows * 32U + threadIdx.x * 8U;
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
        uint32_t i_kpr153 = 0U;
        for (; i_kpr153 < 1U; i_kpr153++) {
            uint32_t j_kpr154 = i_kpr153;
            vec_memcpy(elems_tile0 + (j_kpr154 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr154 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr155 = 0U;
        for (; i_kpr155 < 2U; i_kpr155++) {
            uint32_t j_kpr156 = i_kpr155;
            vec_memcpy(col_ind_tile0 + (j_kpr156 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr156 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr157 = 0U;
        for (; i_kpr157 < to_; i_kpr157++) {
            uint32_t j_kpr158 = i_kpr157;
            elems_tile0[j_kpr158 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    __nv_bfloat16 lchunk[8U];
                    for (uint32_t _i = 0U; _i < 8U; ++_i)
                        lchunk[_i] = __float2bfloat16(0.0f);
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 8U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 8U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 8U; ix++) {
                        uint32_t __anf4 = ix;
                        double __anf3 = out[__anf1 * 8U + ix];
                        __nv_bfloat16 __anf11 = lchunk[ix];
                        out[__anf1 * 8U + __anf4] =
                            __anf3 + (double) __half2float(kx) *
                                         (double) __bfloat162float(__anf11);
                    }
                }
            }
        }
        idx = 1U;
        nnz -= 32U;
        for (; nnz >= 32U; nnz -= 32U) {
            uint32_t off = ri_ + idx * 32U;
            __syncthreads();
            uint32_t i_kpr159 = 0U;
            for (; i_kpr159 < 1U; i_kpr159++) {
                uint32_t j_kpr160 = i_kpr159;
                vec_memcpy(elems_tile0 + (j_kpr160 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr160 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr161 = 0U;
            for (; i_kpr161 < 2U; i_kpr161++) {
                uint32_t j_kpr162 = i_kpr161;
                vec_memcpy(col_ind_tile0 + (j_kpr162 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr162 * 4U + threadIdx.x) * 4U));
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
                        __nv_bfloat16 lchunk[8U];
                        for (uint32_t _i = 0U; _i < 8U; ++_i)
                            lchunk[_i] = __float2bfloat16(0.0f);
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 8U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 8U));
                            k2 = kv1 + 1U;
                        }
                        uint32_t ix = 0U;
                        for (; ix < 8U; ix++) {
                            uint32_t __anf4 = ix;
                            double __anf3 = out[__anf1 * 8U + ix];
                            __nv_bfloat16 __anf11 = lchunk[ix];
                            out[__anf1 * 8U + __anf4] =
                                __anf3 + (double) __half2float(kx) *
                                             (double) __bfloat162float(__anf11);
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
    uint32_t hi_kpr164 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr163 = 0U;
    for (; i_kpr163 < hi_kpr164; i_kpr163++) {
        uint32_t j_kpr165 = i_kpr163;
        elems_tile0[j_kpr165 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr165 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr165 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr165 * 4U + threadIdx.x];
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
                __nv_bfloat16 lchunk[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    lchunk[_i] = __float2bfloat16(0.0f);
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 8U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 8U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 8U; ix++) {
                    uint32_t __anf4 = ix;
                    double __anf3 = out[__anf1 * 8U + ix];
                    __nv_bfloat16 __anf11 = lchunk[ix];
                    out[__anf1 * 8U + __anf4] =
                        __anf3 + (double) __half2float(kx) *
                                     (double) __bfloat162float(__anf11);
                }
            }
        }
    }
    uint32_t i_kpr168 = 0U;
    for (; i_kpr168 < 1U; i_kpr168++) {
        uint32_t j_kpr169 = i_kpr168;
        uint32_t off = n_idx + j_kpr169 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr166 = 0U;
            for (; i_kpr166 < 2U; i_kpr166++) {
                uint32_t j_kpr167 = i_kpr166;
                uint32_t start = j_kpr169 * 8U + j_kpr167 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr167 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = (float) out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr170 = 0U;
        for (; i_kpr170 < 1U; i_kpr170++) {
            uint32_t j_kpr171 = i_kpr170;
            vec_memcpy(elems_tile0 + (j_kpr171 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr171 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr172 = 0U;
        for (; i_kpr172 < 2U; i_kpr172++) {
            uint32_t j_kpr173 = i_kpr172;
            vec_memcpy(col_ind_tile0 + (j_kpr173 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr173 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr174 = 0U;
        for (; i_kpr174 < to_; i_kpr174++) {
            uint32_t j_kpr175 = i_kpr174;
            elems_tile0[j_kpr175 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 4U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr176 = 0U;
            for (; i_kpr176 < 1U; i_kpr176++) {
                uint32_t j_kpr177 = i_kpr176;
                vec_memcpy(elems_tile0 + (j_kpr177 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr177 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr178 = 0U;
            for (; i_kpr178 < 2U; i_kpr178++) {
                uint32_t j_kpr179 = i_kpr178;
                vec_memcpy(col_ind_tile0 + (j_kpr179 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr179 * 4U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 4U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr181 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr180 = 0U;
    for (; i_kpr180 < hi_kpr181; i_kpr180++) {
        uint32_t j_kpr182 = i_kpr180;
        elems_tile0[j_kpr182 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr182 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr182 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr182 * 4U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 4U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
    uint32_t i_kpr185 = 0U;
    for (; i_kpr185 < 2U; i_kpr185++) {
        uint32_t j_kpr186 = i_kpr185;
        uint32_t off = n_idx + j_kpr186 * 4U * 4U;
        if (off < cols) {
            uint32_t i_kpr183 = 0U;
            for (; i_kpr183 < 1U; i_kpr183++) {
                uint32_t j_kpr184 = i_kpr183;
                uint32_t start = j_kpr186 * 4U + j_kpr184 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr184 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = (float) out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr187 = 0U;
        for (; i_kpr187 < 1U; i_kpr187++) {
            uint32_t j_kpr188 = i_kpr187;
            vec_memcpy(elems_tile0 + (j_kpr188 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr188 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr189 = 0U;
        for (; i_kpr189 < 2U; i_kpr189++) {
            uint32_t j_kpr190 = i_kpr189;
            vec_memcpy(col_ind_tile0 + (j_kpr190 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr190 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr191 = 0U;
        for (; i_kpr191 < to_; i_kpr191++) {
            uint32_t j_kpr192 = i_kpr191;
            elems_tile0[j_kpr192 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 4U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr193 = 0U;
            for (; i_kpr193 < 1U; i_kpr193++) {
                uint32_t j_kpr194 = i_kpr193;
                vec_memcpy(elems_tile0 + (j_kpr194 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr194 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr195 = 0U;
            for (; i_kpr195 < 2U; i_kpr195++) {
                uint32_t j_kpr196 = i_kpr195;
                vec_memcpy(col_ind_tile0 + (j_kpr196 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr196 * 4U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 4U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr198 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr197 = 0U;
    for (; i_kpr197 < hi_kpr198; i_kpr197++) {
        uint32_t j_kpr199 = i_kpr197;
        elems_tile0[j_kpr199 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr199 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr199 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr199 * 4U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 4U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
    uint32_t i_kpr202 = 0U;
    for (; i_kpr202 < 2U; i_kpr202++) {
        uint32_t j_kpr203 = i_kpr202;
        uint32_t off = n_idx + j_kpr203 * 4U * 4U;
        if (off < cols) {
            uint32_t i_kpr200 = 0U;
            for (; i_kpr200 < 1U; i_kpr200++) {
                uint32_t j_kpr201 = i_kpr200;
                uint32_t start = j_kpr203 * 4U + j_kpr201 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr201 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = (float) out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr204 = 0U;
        for (; i_kpr204 < 1U; i_kpr204++) {
            uint32_t j_kpr205 = i_kpr204;
            vec_memcpy(elems_tile0 + (j_kpr205 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr205 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr206 = 0U;
        for (; i_kpr206 < 2U; i_kpr206++) {
            uint32_t j_kpr207 = i_kpr206;
            vec_memcpy(col_ind_tile0 + (j_kpr207 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr207 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr208 = 0U;
        for (; i_kpr208 < to_; i_kpr208++) {
            uint32_t j_kpr209 = i_kpr208;
            elems_tile0[j_kpr209 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    uint32_t k2 = 0U;
                    while (k2 < 2U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr210 = 0U;
            for (; i_kpr210 < 1U; i_kpr210++) {
                uint32_t j_kpr211 = i_kpr210;
                vec_memcpy(elems_tile0 + (j_kpr211 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr211 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr212 = 0U;
            for (; i_kpr212 < 2U; i_kpr212++) {
                uint32_t j_kpr213 = i_kpr212;
                vec_memcpy(col_ind_tile0 + (j_kpr213 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr213 * 4U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 2U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr215 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr214 = 0U;
    for (; i_kpr214 < hi_kpr215; i_kpr214++) {
        uint32_t j_kpr216 = i_kpr214;
        elems_tile0[j_kpr216 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr216 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr216 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr216 * 4U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 2U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
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
    uint32_t i_kpr219 = 0U;
    for (; i_kpr219 < 1U; i_kpr219++) {
        uint32_t j_kpr220 = i_kpr219;
        uint32_t off = n_idx + j_kpr220 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr217 = 0U;
            for (; i_kpr217 < 1U; i_kpr217++) {
                uint32_t j_kpr218 = i_kpr217;
                uint32_t start = j_kpr220 * 8U + j_kpr218 * 8U;
                uint32_t address = cols * m_idx + (off + j_kpr218 * 8U);
                half combined[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    combined[_i] = __float2half_rn(0.0f);
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 8U; lane++) {
                    uint32_t __anf4 = lane;
                    float __anf1 = out[start + lane];
                    combined[__anf4] = __float2half_rn(
                        beta * __half2float(combined[lane]) + alpha * __anf1);
                }
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr221 = 0U;
        for (; i_kpr221 < 1U; i_kpr221++) {
            uint32_t j_kpr222 = i_kpr221;
            vec_memcpy(elems_tile0 + (j_kpr222 * 4U + threadIdx.x) * 8U,
                gA.elems + (ri_ + (j_kpr222 * 4U + threadIdx.x) * 8U));
        }
        uint32_t i_kpr223 = 0U;
        for (; i_kpr223 < 2U; i_kpr223++) {
            uint32_t j_kpr224 = i_kpr223;
            vec_memcpy(col_ind_tile0 + (j_kpr224 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr224 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr225 = 0U;
        for (; i_kpr225 < to_; i_kpr225++) {
            uint32_t j_kpr226 = i_kpr225;
            elems_tile0[j_kpr226 * 4U + threadIdx.x] = __float2half_rn(0.0f);
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
                    uint32_t k2 = 0U;
                    while (k2 < 2U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 8U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr227 = 0U;
            for (; i_kpr227 < 1U; i_kpr227++) {
                uint32_t j_kpr228 = i_kpr227;
                vec_memcpy(elems_tile0 + (j_kpr228 * 4U + threadIdx.x) * 8U,
                    gA.elems + (off + (j_kpr228 * 4U + threadIdx.x) * 8U));
            }
            uint32_t i_kpr229 = 0U;
            for (; i_kpr229 < 2U; i_kpr229++) {
                uint32_t j_kpr230 = i_kpr229;
                vec_memcpy(col_ind_tile0 + (j_kpr230 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr230 * 4U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 2U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 8U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr232 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr231 = 0U;
    for (; i_kpr231 < hi_kpr232; i_kpr231++) {
        uint32_t j_kpr233 = i_kpr231;
        elems_tile0[j_kpr233 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr233 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr233 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr233 * 4U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 2U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 8U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
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
    uint32_t i_kpr236 = 0U;
    for (; i_kpr236 < 1U; i_kpr236++) {
        uint32_t j_kpr237 = i_kpr236;
        uint32_t off = n_idx + j_kpr237 * 4U * 8U;
        if (off < cols) {
            uint32_t i_kpr234 = 0U;
            for (; i_kpr234 < 1U; i_kpr234++) {
                uint32_t j_kpr235 = i_kpr234;
                uint32_t start = j_kpr237 * 8U + j_kpr235 * 8U;
                uint32_t address = cols * m_idx + (off + j_kpr235 * 8U);
                half combined[8U];
                for (uint32_t _i = 0U; _i < 8U; ++_i)
                    combined[_i] = __float2half_rn(0.0f);
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 8U; lane++) {
                    uint32_t __anf4 = lane;
                    float __anf1 = out[start + lane];
                    combined[__anf4] = __float2half_rn(
                        beta * __half2float(combined[lane]) + alpha * __anf1);
                }
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr238 = 0U;
        for (; i_kpr238 < 8U; i_kpr238++) {
            uint32_t j_kpr239 = i_kpr238;
            vec_memcpy(elems_tile0 + (j_kpr239 + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr239 + threadIdx.x) * 4U));
        }
        uint32_t i_kpr240 = 0U;
        for (; i_kpr240 < 8U; i_kpr240++) {
            uint32_t j_kpr241 = i_kpr240;
            vec_memcpy(col_ind_tile0 + (j_kpr241 + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr241 + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i_kpr242 = 0U;
        for (; i_kpr242 < to_; i_kpr242++)
            elems_tile0[i_kpr242 + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr244 = 0U;
            for (; i_kpr244 < 8U; i_kpr244++) {
                uint32_t j_kpr245 = i_kpr244;
                vec_memcpy(elems_tile0 + (j_kpr245 + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr245 + threadIdx.x) * 4U));
            }
            uint32_t i_kpr246 = 0U;
            for (; i_kpr246 < 8U; i_kpr246++) {
                uint32_t j_kpr247 = i_kpr246;
                vec_memcpy(col_ind_tile0 + (j_kpr247 + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr247 + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr249 = __anf01 - threadIdx.x;
    uint32_t i_kpr248 = 0U;
    for (; i_kpr248 < hi_kpr249; i_kpr248++) {
        uint32_t j_kpr250 = i_kpr248;
        elems_tile0[j_kpr250 + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr250 + threadIdx.x];
        col_ind_tile0[j_kpr250 + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr250 + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U) + kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr253 = 0U;
    for (; i_kpr253 < 1U; i_kpr253++) {
        uint32_t j_kpr254 = i_kpr253;
        uint32_t off = n_idx + j_kpr254 * 4U;
        if (off < cols) {
            uint32_t i_kpr251 = 0U;
            for (; i_kpr251 < 1U; i_kpr251++) {
                uint32_t j_kpr252 = i_kpr251;
                uint32_t start = j_kpr254 * 4U + j_kpr252 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr252 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr255 = 0U;
        for (; i_kpr255 < 4U; i_kpr255++) {
            uint32_t j_kpr256 = i_kpr255;
            vec_memcpy(elems_tile0 + (j_kpr256 * 2U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr256 * 2U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr257 = 0U;
        for (; i_kpr257 < 4U; i_kpr257++) {
            uint32_t j_kpr258 = i_kpr257;
            vec_memcpy(col_ind_tile0 + (j_kpr258 * 2U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr258 * 2U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i_kpr259 = 0U;
        for (; i_kpr259 < to_; i_kpr259++)
            elems_tile0[i_kpr259 * 2U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 2U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            for (; i_kpr261 < 4U; i_kpr261++) {
                uint32_t j_kpr262 = i_kpr261;
                vec_memcpy(elems_tile0 + (j_kpr262 * 2U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr262 * 2U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr263 = 0U;
            for (; i_kpr263 < 4U; i_kpr263++) {
                uint32_t j_kpr264 = i_kpr263;
                vec_memcpy(col_ind_tile0 + (j_kpr264 * 2U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr264 * 2U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 2U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr266 = (__anf01 + 1U - threadIdx.x) / 2U;
    uint32_t i_kpr265 = 0U;
    for (; i_kpr265 < hi_kpr266; i_kpr265++) {
        uint32_t j_kpr267 = i_kpr265;
        elems_tile0[j_kpr267 * 2U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr267 * 2U + threadIdx.x];
        col_ind_tile0[j_kpr267 * 2U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr267 * 2U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 2U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr270 = 0U;
    for (; i_kpr270 < 1U; i_kpr270++) {
        uint32_t j_kpr271 = i_kpr270;
        uint32_t off = n_idx + j_kpr271 * 2U * 4U;
        if (off < cols) {
            uint32_t i_kpr268 = 0U;
            for (; i_kpr268 < 1U; i_kpr268++) {
                uint32_t j_kpr269 = i_kpr268;
                uint32_t start = j_kpr271 * 4U + j_kpr269 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr269 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr272 = 0U;
        for (; i_kpr272 < 2U; i_kpr272++) {
            uint32_t j_kpr273 = i_kpr272;
            vec_memcpy(elems_tile0 + (j_kpr273 * 4U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr273 * 4U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr274 = 0U;
        for (; i_kpr274 < 2U; i_kpr274++) {
            uint32_t j_kpr275 = i_kpr274;
            vec_memcpy(col_ind_tile0 + (j_kpr275 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr275 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr276 = 0U;
        for (; i_kpr276 < to_; i_kpr276++)
            elems_tile0[i_kpr276 * 4U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr278 = 0U;
            for (; i_kpr278 < 2U; i_kpr278++) {
                uint32_t j_kpr279 = i_kpr278;
                vec_memcpy(elems_tile0 + (j_kpr279 * 4U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr279 * 4U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr280 = 0U;
            for (; i_kpr280 < 2U; i_kpr280++) {
                uint32_t j_kpr281 = i_kpr280;
                vec_memcpy(col_ind_tile0 + (j_kpr281 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr281 * 4U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr283 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr282 = 0U;
    for (; i_kpr282 < hi_kpr283; i_kpr282++) {
        uint32_t j_kpr284 = i_kpr282;
        elems_tile0[j_kpr284 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr284 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr284 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr284 * 4U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr287 = 0U;
    for (; i_kpr287 < 1U; i_kpr287++) {
        uint32_t j_kpr288 = i_kpr287;
        uint32_t off = n_idx + j_kpr288 * 4U * 4U;
        if (off < cols) {
            uint32_t i_kpr285 = 0U;
            for (; i_kpr285 < 1U; i_kpr285++) {
                uint32_t j_kpr286 = i_kpr285;
                uint32_t start = j_kpr288 * 4U + j_kpr286 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr286 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr289 = 0U;
        for (; i_kpr289 < 1U; i_kpr289++) {
            uint32_t j_kpr290 = i_kpr289;
            vec_memcpy(elems_tile0 + (j_kpr290 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr290 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr291 = 0U;
        for (; i_kpr291 < 1U; i_kpr291++) {
            uint32_t j_kpr292 = i_kpr291;
            vec_memcpy(col_ind_tile0 + (j_kpr292 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr292 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr293 = 0U;
        for (; i_kpr293 < to_; i_kpr293++)
            elems_tile0[i_kpr293 * 8U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 8U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr295 = 0U;
            for (; i_kpr295 < 1U; i_kpr295++) {
                uint32_t j_kpr296 = i_kpr295;
                vec_memcpy(elems_tile0 + (j_kpr296 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr296 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr297 = 0U;
            for (; i_kpr297 < 1U; i_kpr297++) {
                uint32_t j_kpr298 = i_kpr297;
                vec_memcpy(col_ind_tile0 + (j_kpr298 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr298 * 8U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 8U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr300 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr299 = 0U;
    for (; i_kpr299 < hi_kpr300; i_kpr299++) {
        uint32_t j_kpr301 = i_kpr299;
        elems_tile0[j_kpr301 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr301 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr301 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr301 * 8U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 8U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr304 = 0U;
    for (; i_kpr304 < 1U; i_kpr304++) {
        uint32_t j_kpr305 = i_kpr304;
        uint32_t off = n_idx + j_kpr305 * 8U * 4U;
        if (off < cols) {
            uint32_t i_kpr302 = 0U;
            for (; i_kpr302 < 1U; i_kpr302++) {
                uint32_t j_kpr303 = i_kpr302;
                uint32_t start = j_kpr305 * 4U + j_kpr303 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr303 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr306 = 0U;
        for (; i_kpr306 < 1U; i_kpr306++) {
            uint32_t j_kpr307 = i_kpr306;
            vec_memcpy(elems_tile0 + (j_kpr307 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr307 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr308 = 0U;
        for (; i_kpr308 < 1U; i_kpr308++) {
            uint32_t j_kpr309 = i_kpr308;
            vec_memcpy(col_ind_tile0 + (j_kpr309 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr309 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr310 = 0U;
        for (; i_kpr310 < to_; i_kpr310++)
            elems_tile0[i_kpr310 * 8U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 8U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr312 = 0U;
            for (; i_kpr312 < 1U; i_kpr312++) {
                uint32_t j_kpr313 = i_kpr312;
                vec_memcpy(elems_tile0 + (j_kpr313 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr313 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr314 = 0U;
            for (; i_kpr314 < 1U; i_kpr314++) {
                uint32_t j_kpr315 = i_kpr314;
                vec_memcpy(col_ind_tile0 + (j_kpr315 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr315 * 8U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 8U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr317 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr316 = 0U;
    for (; i_kpr316 < hi_kpr317; i_kpr316++) {
        uint32_t j_kpr318 = i_kpr316;
        elems_tile0[j_kpr318 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr318 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr318 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr318 * 8U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 8U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr321 = 0U;
    for (; i_kpr321 < 2U; i_kpr321++) {
        uint32_t j_kpr322 = i_kpr321;
        uint32_t off = n_idx + j_kpr322 * 8U * 4U;
        if (off < cols) {
            uint32_t i_kpr319 = 0U;
            for (; i_kpr319 < 1U; i_kpr319++) {
                uint32_t j_kpr320 = i_kpr319;
                uint32_t start = j_kpr322 * 4U + j_kpr320 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr320 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr323 = 0U;
        for (; i_kpr323 < 8U; i_kpr323++) {
            uint32_t j_kpr324 = i_kpr323;
            vec_memcpy(elems_tile0 + (j_kpr324 + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr324 + threadIdx.x) * 4U));
        }
        uint32_t i_kpr325 = 0U;
        for (; i_kpr325 < 8U; i_kpr325++) {
            uint32_t j_kpr326 = i_kpr325;
            vec_memcpy(col_ind_tile0 + (j_kpr326 + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr326 + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = ri - ri_ - threadIdx.x;
        uint32_t i_kpr327 = 0U;
        for (; i_kpr327 < to_; i_kpr327++)
            elems_tile0[i_kpr327 + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr329 = 0U;
            for (; i_kpr329 < 8U; i_kpr329++) {
                uint32_t j_kpr330 = i_kpr329;
                vec_memcpy(elems_tile0 + (j_kpr330 + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr330 + threadIdx.x) * 4U));
            }
            uint32_t i_kpr331 = 0U;
            for (; i_kpr331 < 8U; i_kpr331++) {
                uint32_t j_kpr332 = i_kpr331;
                vec_memcpy(col_ind_tile0 + (j_kpr332 + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr332 + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr334 = __anf01 - threadIdx.x;
    uint32_t i_kpr333 = 0U;
    for (; i_kpr333 < hi_kpr334; i_kpr333++) {
        uint32_t j_kpr335 = i_kpr333;
        elems_tile0[j_kpr335 + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr335 + threadIdx.x];
        col_ind_tile0[j_kpr335 + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr335 + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U) + kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr338 = 0U;
    for (; i_kpr338 < 1U; i_kpr338++) {
        uint32_t j_kpr339 = i_kpr338;
        uint32_t off = n_idx + j_kpr339 * 4U;
        if (off < cols) {
            uint32_t i_kpr336 = 0U;
            for (; i_kpr336 < 1U; i_kpr336++) {
                uint32_t j_kpr337 = i_kpr336;
                uint32_t start = j_kpr339 * 4U + j_kpr337 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr337 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr340 = 0U;
        for (; i_kpr340 < 4U; i_kpr340++) {
            uint32_t j_kpr341 = i_kpr340;
            vec_memcpy(elems_tile0 + (j_kpr341 * 2U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr341 * 2U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr342 = 0U;
        for (; i_kpr342 < 4U; i_kpr342++) {
            uint32_t j_kpr343 = i_kpr342;
            vec_memcpy(col_ind_tile0 + (j_kpr343 * 2U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr343 * 2U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 1U - threadIdx.x) / 2U;
        uint32_t i_kpr344 = 0U;
        for (; i_kpr344 < to_; i_kpr344++)
            elems_tile0[i_kpr344 * 2U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 2U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr346 = 0U;
            for (; i_kpr346 < 4U; i_kpr346++) {
                uint32_t j_kpr347 = i_kpr346;
                vec_memcpy(elems_tile0 + (j_kpr347 * 2U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr347 * 2U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr348 = 0U;
            for (; i_kpr348 < 4U; i_kpr348++) {
                uint32_t j_kpr349 = i_kpr348;
                vec_memcpy(col_ind_tile0 + (j_kpr349 * 2U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr349 * 2U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 2U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr351 = (__anf01 + 1U - threadIdx.x) / 2U;
    uint32_t i_kpr350 = 0U;
    for (; i_kpr350 < hi_kpr351; i_kpr350++) {
        uint32_t j_kpr352 = i_kpr350;
        elems_tile0[j_kpr352 * 2U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr352 * 2U + threadIdx.x];
        col_ind_tile0[j_kpr352 * 2U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr352 * 2U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 2U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr355 = 0U;
    for (; i_kpr355 < 1U; i_kpr355++) {
        uint32_t j_kpr356 = i_kpr355;
        uint32_t off = n_idx + j_kpr356 * 2U * 4U;
        if (off < cols) {
            uint32_t i_kpr353 = 0U;
            for (; i_kpr353 < 1U; i_kpr353++) {
                uint32_t j_kpr354 = i_kpr353;
                uint32_t start = j_kpr356 * 4U + j_kpr354 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr354 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr357 = 0U;
        for (; i_kpr357 < 2U; i_kpr357++) {
            uint32_t j_kpr358 = i_kpr357;
            vec_memcpy(elems_tile0 + (j_kpr358 * 4U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr358 * 4U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr359 = 0U;
        for (; i_kpr359 < 2U; i_kpr359++) {
            uint32_t j_kpr360 = i_kpr359;
            vec_memcpy(col_ind_tile0 + (j_kpr360 * 4U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr360 * 4U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 3U - threadIdx.x) / 4U;
        uint32_t i_kpr361 = 0U;
        for (; i_kpr361 < to_; i_kpr361++)
            elems_tile0[i_kpr361 * 4U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 4U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr363 = 0U;
            for (; i_kpr363 < 2U; i_kpr363++) {
                uint32_t j_kpr364 = i_kpr363;
                vec_memcpy(elems_tile0 + (j_kpr364 * 4U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr364 * 4U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr365 = 0U;
            for (; i_kpr365 < 2U; i_kpr365++) {
                uint32_t j_kpr366 = i_kpr365;
                vec_memcpy(col_ind_tile0 + (j_kpr366 * 4U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr366 * 4U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 4U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr368 = (__anf01 + 3U - threadIdx.x) / 4U;
    uint32_t i_kpr367 = 0U;
    for (; i_kpr367 < hi_kpr368; i_kpr367++) {
        uint32_t j_kpr369 = i_kpr367;
        elems_tile0[j_kpr369 * 4U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr369 * 4U + threadIdx.x];
        col_ind_tile0[j_kpr369 * 4U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr369 * 4U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 4U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr372 = 0U;
    for (; i_kpr372 < 1U; i_kpr372++) {
        uint32_t j_kpr373 = i_kpr372;
        uint32_t off = n_idx + j_kpr373 * 4U * 4U;
        if (off < cols) {
            uint32_t i_kpr370 = 0U;
            for (; i_kpr370 < 1U; i_kpr370++) {
                uint32_t j_kpr371 = i_kpr370;
                uint32_t start = j_kpr373 * 4U + j_kpr371 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr371 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr374 = 0U;
        for (; i_kpr374 < 1U; i_kpr374++) {
            uint32_t j_kpr375 = i_kpr374;
            vec_memcpy(elems_tile0 + (j_kpr375 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr375 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr376 = 0U;
        for (; i_kpr376 < 1U; i_kpr376++) {
            uint32_t j_kpr377 = i_kpr376;
            vec_memcpy(col_ind_tile0 + (j_kpr377 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr377 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr378 = 0U;
        for (; i_kpr378 < to_; i_kpr378++)
            elems_tile0[i_kpr378 * 8U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 8U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr380 = 0U;
            for (; i_kpr380 < 1U; i_kpr380++) {
                uint32_t j_kpr381 = i_kpr380;
                vec_memcpy(elems_tile0 + (j_kpr381 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr381 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr382 = 0U;
            for (; i_kpr382 < 1U; i_kpr382++) {
                uint32_t j_kpr383 = i_kpr382;
                vec_memcpy(col_ind_tile0 + (j_kpr383 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr383 * 8U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 8U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr385 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr384 = 0U;
    for (; i_kpr384 < hi_kpr385; i_kpr384++) {
        uint32_t j_kpr386 = i_kpr384;
        elems_tile0[j_kpr386 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr386 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr386 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr386 * 8U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 8U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr389 = 0U;
    for (; i_kpr389 < 1U; i_kpr389++) {
        uint32_t j_kpr390 = i_kpr389;
        uint32_t off = n_idx + j_kpr390 * 8U * 4U;
        if (off < cols) {
            uint32_t i_kpr387 = 0U;
            for (; i_kpr387 < 1U; i_kpr387++) {
                uint32_t j_kpr388 = i_kpr387;
                uint32_t start = j_kpr390 * 4U + j_kpr388 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr388 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr391 = 0U;
        for (; i_kpr391 < 1U; i_kpr391++) {
            uint32_t j_kpr392 = i_kpr391;
            vec_memcpy(elems_tile0 + (j_kpr392 * 8U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr392 * 8U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr393 = 0U;
        for (; i_kpr393 < 1U; i_kpr393++) {
            uint32_t j_kpr394 = i_kpr393;
            vec_memcpy(col_ind_tile0 + (j_kpr394 * 8U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr394 * 8U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 7U - threadIdx.x) / 8U;
        uint32_t i_kpr395 = 0U;
        for (; i_kpr395 < to_; i_kpr395++)
            elems_tile0[i_kpr395 * 8U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 8U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr397 = 0U;
            for (; i_kpr397 < 1U; i_kpr397++) {
                uint32_t j_kpr398 = i_kpr397;
                vec_memcpy(elems_tile0 + (j_kpr398 * 8U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr398 * 8U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr399 = 0U;
            for (; i_kpr399 < 1U; i_kpr399++) {
                uint32_t j_kpr400 = i_kpr399;
                vec_memcpy(col_ind_tile0 + (j_kpr400 * 8U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr400 * 8U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 8U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr402 = (__anf01 + 7U - threadIdx.x) / 8U;
    uint32_t i_kpr401 = 0U;
    for (; i_kpr401 < hi_kpr402; i_kpr401++) {
        uint32_t j_kpr403 = i_kpr401;
        elems_tile0[j_kpr403 * 8U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr403 * 8U + threadIdx.x];
        col_ind_tile0[j_kpr403 * 8U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr403 * 8U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 8U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr406 = 0U;
    for (; i_kpr406 < 2U; i_kpr406++) {
        uint32_t j_kpr407 = i_kpr406;
        uint32_t off = n_idx + j_kpr407 * 8U * 4U;
        if (off < cols) {
            uint32_t i_kpr404 = 0U;
            for (; i_kpr404 < 1U; i_kpr404++) {
                uint32_t j_kpr405 = i_kpr404;
                uint32_t start = j_kpr407 * 4U + j_kpr405 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr405 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr408 = 0U;
        for (; i_kpr408 < 1U; i_kpr408++) {
            uint32_t j_kpr409 = i_kpr408;
            vec_memcpy(elems_tile0 + (j_kpr409 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr409 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr410 = 0U;
        for (; i_kpr410 < 1U; i_kpr410++) {
            uint32_t j_kpr411 = i_kpr410;
            vec_memcpy(col_ind_tile0 + (j_kpr411 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr411 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr412 = 0U;
        for (; i_kpr412 < to_; i_kpr412++)
            elems_tile0[i_kpr412 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr414 = 0U;
            for (; i_kpr414 < 1U; i_kpr414++) {
                uint32_t j_kpr415 = i_kpr414;
                vec_memcpy(elems_tile0 + (j_kpr415 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr415 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr416 = 0U;
            for (; i_kpr416 < 1U; i_kpr416++) {
                uint32_t j_kpr417 = i_kpr416;
                vec_memcpy(col_ind_tile0 + (j_kpr417 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr417 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr419 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr418 = 0U;
    for (; i_kpr418 < hi_kpr419; i_kpr418++) {
        uint32_t j_kpr420 = i_kpr418;
        elems_tile0[j_kpr420 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr420 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr420 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr420 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr423 = 0U;
    for (; i_kpr423 < 1U; i_kpr423++) {
        uint32_t j_kpr424 = i_kpr423;
        uint32_t off = n_idx + j_kpr424 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr421 = 0U;
            for (; i_kpr421 < 1U; i_kpr421++) {
                uint32_t j_kpr422 = i_kpr421;
                uint32_t start = j_kpr424 * 4U + j_kpr422 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr422 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr425 = 0U;
        for (; i_kpr425 < 1U; i_kpr425++) {
            uint32_t j_kpr426 = i_kpr425;
            vec_memcpy(elems_tile0 + (j_kpr426 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr426 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr427 = 0U;
        for (; i_kpr427 < 1U; i_kpr427++) {
            uint32_t j_kpr428 = i_kpr427;
            vec_memcpy(col_ind_tile0 + (j_kpr428 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr428 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr429 = 0U;
        for (; i_kpr429 < to_; i_kpr429++)
            elems_tile0[i_kpr429 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr431 = 0U;
            for (; i_kpr431 < 1U; i_kpr431++) {
                uint32_t j_kpr432 = i_kpr431;
                vec_memcpy(elems_tile0 + (j_kpr432 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr432 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr433 = 0U;
            for (; i_kpr433 < 1U; i_kpr433++) {
                uint32_t j_kpr434 = i_kpr433;
                vec_memcpy(col_ind_tile0 + (j_kpr434 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr434 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr436 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr435 = 0U;
    for (; i_kpr435 < hi_kpr436; i_kpr435++) {
        uint32_t j_kpr437 = i_kpr435;
        elems_tile0[j_kpr437 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr437 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr437 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr437 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr440 = 0U;
    for (; i_kpr440 < 1U; i_kpr440++) {
        uint32_t j_kpr441 = i_kpr440;
        uint32_t off = n_idx + j_kpr441 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr438 = 0U;
            for (; i_kpr438 < 1U; i_kpr438++) {
                uint32_t j_kpr439 = i_kpr438;
                uint32_t start = j_kpr441 * 4U + j_kpr439 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr439 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr442 = 0U;
        for (; i_kpr442 < 1U; i_kpr442++) {
            uint32_t j_kpr443 = i_kpr442;
            vec_memcpy(elems_tile0 + (j_kpr443 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr443 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr444 = 0U;
        for (; i_kpr444 < 1U; i_kpr444++) {
            uint32_t j_kpr445 = i_kpr444;
            vec_memcpy(col_ind_tile0 + (j_kpr445 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr445 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr446 = 0U;
        for (; i_kpr446 < to_; i_kpr446++)
            elems_tile0[i_kpr446 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr448 = 0U;
            for (; i_kpr448 < 1U; i_kpr448++) {
                uint32_t j_kpr449 = i_kpr448;
                vec_memcpy(elems_tile0 + (j_kpr449 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr449 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr450 = 0U;
            for (; i_kpr450 < 1U; i_kpr450++) {
                uint32_t j_kpr451 = i_kpr450;
                vec_memcpy(col_ind_tile0 + (j_kpr451 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr451 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr453 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr452 = 0U;
    for (; i_kpr452 < hi_kpr453; i_kpr452++) {
        uint32_t j_kpr454 = i_kpr452;
        elems_tile0[j_kpr454 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr454 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr454 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr454 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr457 = 0U;
    for (; i_kpr457 < 2U; i_kpr457++) {
        uint32_t j_kpr458 = i_kpr457;
        uint32_t off = n_idx + j_kpr458 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr455 = 0U;
            for (; i_kpr455 < 1U; i_kpr455++) {
                uint32_t j_kpr456 = i_kpr455;
                uint32_t start = j_kpr458 * 4U + j_kpr456 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr456 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr459 = 0U;
        for (; i_kpr459 < 1U; i_kpr459++) {
            uint32_t j_kpr460 = i_kpr459;
            vec_memcpy(elems_tile0 + (j_kpr460 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr460 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr461 = 0U;
        for (; i_kpr461 < 1U; i_kpr461++) {
            uint32_t j_kpr462 = i_kpr461;
            vec_memcpy(col_ind_tile0 + (j_kpr462 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr462 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr463 = 0U;
        for (; i_kpr463 < to_; i_kpr463++)
            elems_tile0[i_kpr463 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr465 = 0U;
            for (; i_kpr465 < 1U; i_kpr465++) {
                uint32_t j_kpr466 = i_kpr465;
                vec_memcpy(elems_tile0 + (j_kpr466 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr466 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr467 = 0U;
            for (; i_kpr467 < 1U; i_kpr467++) {
                uint32_t j_kpr468 = i_kpr467;
                vec_memcpy(col_ind_tile0 + (j_kpr468 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr468 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr470 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr469 = 0U;
    for (; i_kpr469 < hi_kpr470; i_kpr469++) {
        uint32_t j_kpr471 = i_kpr469;
        elems_tile0[j_kpr471 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr471 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr471 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr471 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr474 = 0U;
    for (; i_kpr474 < 2U; i_kpr474++) {
        uint32_t j_kpr475 = i_kpr474;
        uint32_t off = n_idx + j_kpr475 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr472 = 0U;
            for (; i_kpr472 < 1U; i_kpr472++) {
                uint32_t j_kpr473 = i_kpr472;
                uint32_t start = j_kpr475 * 4U + j_kpr473 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr473 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr476 = 0U;
        for (; i_kpr476 < 1U; i_kpr476++) {
            uint32_t j_kpr477 = i_kpr476;
            vec_memcpy(elems_tile0 + (j_kpr477 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr477 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr478 = 0U;
        for (; i_kpr478 < 1U; i_kpr478++) {
            uint32_t j_kpr479 = i_kpr478;
            vec_memcpy(col_ind_tile0 + (j_kpr479 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr479 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr480 = 0U;
        for (; i_kpr480 < to_; i_kpr480++)
            elems_tile0[i_kpr480 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr482 = 0U;
            for (; i_kpr482 < 1U; i_kpr482++) {
                uint32_t j_kpr483 = i_kpr482;
                vec_memcpy(elems_tile0 + (j_kpr483 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr483 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr484 = 0U;
            for (; i_kpr484 < 1U; i_kpr484++) {
                uint32_t j_kpr485 = i_kpr484;
                vec_memcpy(col_ind_tile0 + (j_kpr485 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr485 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr487 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr486 = 0U;
    for (; i_kpr486 < hi_kpr487; i_kpr486++) {
        uint32_t j_kpr488 = i_kpr486;
        elems_tile0[j_kpr488 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr488 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr488 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr488 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr491 = 0U;
    for (; i_kpr491 < 4U; i_kpr491++) {
        uint32_t j_kpr492 = i_kpr491;
        uint32_t off = n_idx + j_kpr492 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr489 = 0U;
            for (; i_kpr489 < 1U; i_kpr489++) {
                uint32_t j_kpr490 = i_kpr489;
                uint32_t start = j_kpr492 * 4U + j_kpr490 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr490 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr493 = 0U;
        for (; i_kpr493 < 1U; i_kpr493++) {
            uint32_t j_kpr494 = i_kpr493;
            vec_memcpy(elems_tile0 + (j_kpr494 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr494 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr495 = 0U;
        for (; i_kpr495 < 1U; i_kpr495++) {
            uint32_t j_kpr496 = i_kpr495;
            vec_memcpy(col_ind_tile0 + (j_kpr496 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr496 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr497 = 0U;
        for (; i_kpr497 < to_; i_kpr497++)
            elems_tile0[i_kpr497 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr499 = 0U;
            for (; i_kpr499 < 1U; i_kpr499++) {
                uint32_t j_kpr500 = i_kpr499;
                vec_memcpy(elems_tile0 + (j_kpr500 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr500 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr501 = 0U;
            for (; i_kpr501 < 1U; i_kpr501++) {
                uint32_t j_kpr502 = i_kpr501;
                vec_memcpy(col_ind_tile0 + (j_kpr502 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr502 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr504 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr503 = 0U;
    for (; i_kpr503 < hi_kpr504; i_kpr503++) {
        uint32_t j_kpr505 = i_kpr503;
        elems_tile0[j_kpr505 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr505 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr505 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr505 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr508 = 0U;
    for (; i_kpr508 < 4U; i_kpr508++) {
        uint32_t j_kpr509 = i_kpr508;
        uint32_t off = n_idx + j_kpr509 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr506 = 0U;
            for (; i_kpr506 < 1U; i_kpr506++) {
                uint32_t j_kpr507 = i_kpr506;
                uint32_t start = j_kpr509 * 4U + j_kpr507 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr507 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr510 = 0U;
        for (; i_kpr510 < 1U; i_kpr510++) {
            uint32_t j_kpr511 = i_kpr510;
            vec_memcpy(elems_tile0 + (j_kpr511 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr511 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr512 = 0U;
        for (; i_kpr512 < 1U; i_kpr512++) {
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr516 = 0U;
            for (; i_kpr516 < 1U; i_kpr516++) {
                uint32_t j_kpr517 = i_kpr516;
                vec_memcpy(elems_tile0 + (j_kpr517 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr517 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr518 = 0U;
            for (; i_kpr518 < 1U; i_kpr518++) {
                uint32_t j_kpr519 = i_kpr518;
                vec_memcpy(col_ind_tile0 + (j_kpr519 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr519 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
            for (; k1 < 8U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 16U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr525 = 0U;
    for (; i_kpr525 < 8U; i_kpr525++) {
        uint32_t j_kpr526 = i_kpr525;
        uint32_t off = n_idx + j_kpr526 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr523 = 0U;
            for (; i_kpr523 < 1U; i_kpr523++) {
                uint32_t j_kpr524 = i_kpr523;
                uint32_t start = j_kpr526 * 4U + j_kpr524 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr524 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr527 = 0U;
        for (; i_kpr527 < 1U; i_kpr527++) {
            uint32_t j_kpr528 = i_kpr527;
            vec_memcpy(elems_tile0 + (j_kpr528 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr528 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr529 = 0U;
        for (; i_kpr529 < 1U; i_kpr529++) {
            uint32_t j_kpr530 = i_kpr529;
            vec_memcpy(col_ind_tile0 + (j_kpr530 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr530 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr531 = 0U;
        for (; i_kpr531 < to_; i_kpr531++)
            elems_tile0[i_kpr531 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr533 = 0U;
            for (; i_kpr533 < 1U; i_kpr533++) {
                uint32_t j_kpr534 = i_kpr533;
                vec_memcpy(elems_tile0 + (j_kpr534 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr534 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr535 = 0U;
            for (; i_kpr535 < 1U; i_kpr535++) {
                uint32_t j_kpr536 = i_kpr535;
                vec_memcpy(col_ind_tile0 + (j_kpr536 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr536 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr538 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr537 = 0U;
    for (; i_kpr537 < hi_kpr538; i_kpr537++) {
        uint32_t j_kpr539 = i_kpr537;
        elems_tile0[j_kpr539 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr539 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr539 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr539 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr542 = 0U;
    for (; i_kpr542 < 8U; i_kpr542++) {
        uint32_t j_kpr543 = i_kpr542;
        uint32_t off = n_idx + j_kpr543 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr540 = 0U;
            for (; i_kpr540 < 1U; i_kpr540++) {
                uint32_t j_kpr541 = i_kpr540;
                uint32_t start = j_kpr543 * 4U + j_kpr541 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr541 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr544 = 0U;
        for (; i_kpr544 < 2U; i_kpr544++) {
            uint32_t j_kpr545 = i_kpr544;
            vec_memcpy(elems_tile0 + (j_kpr545 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr545 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr546 = 0U;
        for (; i_kpr546 < 2U; i_kpr546++) {
            uint32_t j_kpr547 = i_kpr546;
            vec_memcpy(col_ind_tile0 + (j_kpr547 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr547 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr548 = 0U;
        for (; i_kpr548 < to_; i_kpr548++)
            elems_tile0[i_kpr548 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr550 = 0U;
            for (; i_kpr550 < 2U; i_kpr550++) {
                uint32_t j_kpr551 = i_kpr550;
                vec_memcpy(elems_tile0 + (j_kpr551 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr551 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr552 = 0U;
            for (; i_kpr552 < 2U; i_kpr552++) {
                uint32_t j_kpr553 = i_kpr552;
                vec_memcpy(col_ind_tile0 + (j_kpr553 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr553 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr555 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr554 = 0U;
    for (; i_kpr554 < hi_kpr555; i_kpr554++) {
        uint32_t j_kpr556 = i_kpr554;
        elems_tile0[j_kpr556 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr556 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr556 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr556 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr559 = 0U;
    for (; i_kpr559 < 1U; i_kpr559++) {
        uint32_t j_kpr560 = i_kpr559;
        uint32_t off = n_idx + j_kpr560 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr557 = 0U;
            for (; i_kpr557 < 1U; i_kpr557++) {
                uint32_t j_kpr558 = i_kpr557;
                uint32_t start = j_kpr560 * 4U + j_kpr558 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr558 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr561 = 0U;
        for (; i_kpr561 < 2U; i_kpr561++) {
            uint32_t j_kpr562 = i_kpr561;
            vec_memcpy(elems_tile0 + (j_kpr562 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr562 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr563 = 0U;
        for (; i_kpr563 < 2U; i_kpr563++) {
            uint32_t j_kpr564 = i_kpr563;
            vec_memcpy(col_ind_tile0 + (j_kpr564 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr564 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr565 = 0U;
        for (; i_kpr565 < to_; i_kpr565++)
            elems_tile0[i_kpr565 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr567 = 0U;
            for (; i_kpr567 < 2U; i_kpr567++) {
                uint32_t j_kpr568 = i_kpr567;
                vec_memcpy(elems_tile0 + (j_kpr568 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr568 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr569 = 0U;
            for (; i_kpr569 < 2U; i_kpr569++) {
                uint32_t j_kpr570 = i_kpr569;
                vec_memcpy(col_ind_tile0 + (j_kpr570 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr570 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr572 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr571 = 0U;
    for (; i_kpr571 < hi_kpr572; i_kpr571++) {
        uint32_t j_kpr573 = i_kpr571;
        elems_tile0[j_kpr573 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr573 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr573 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr573 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr576 = 0U;
    for (; i_kpr576 < 1U; i_kpr576++) {
        uint32_t j_kpr577 = i_kpr576;
        uint32_t off = n_idx + j_kpr577 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr574 = 0U;
            for (; i_kpr574 < 1U; i_kpr574++) {
                uint32_t j_kpr575 = i_kpr574;
                uint32_t start = j_kpr577 * 4U + j_kpr575 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr575 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr578 = 0U;
        for (; i_kpr578 < 2U; i_kpr578++) {
            uint32_t j_kpr579 = i_kpr578;
            vec_memcpy(elems_tile0 + (j_kpr579 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr579 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr580 = 0U;
        for (; i_kpr580 < 2U; i_kpr580++) {
            uint32_t j_kpr581 = i_kpr580;
            vec_memcpy(col_ind_tile0 + (j_kpr581 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr581 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr582 = 0U;
        for (; i_kpr582 < to_; i_kpr582++)
            elems_tile0[i_kpr582 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr584 = 0U;
            for (; i_kpr584 < 2U; i_kpr584++) {
                uint32_t j_kpr585 = i_kpr584;
                vec_memcpy(elems_tile0 + (j_kpr585 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr585 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr586 = 0U;
            for (; i_kpr586 < 2U; i_kpr586++) {
                uint32_t j_kpr587 = i_kpr586;
                vec_memcpy(col_ind_tile0 + (j_kpr587 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr587 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr589 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr588 = 0U;
    for (; i_kpr588 < hi_kpr589; i_kpr588++) {
        uint32_t j_kpr590 = i_kpr588;
        elems_tile0[j_kpr590 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr590 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr590 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr590 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr593 = 0U;
    for (; i_kpr593 < 2U; i_kpr593++) {
        uint32_t j_kpr594 = i_kpr593;
        uint32_t off = n_idx + j_kpr594 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr591 = 0U;
            for (; i_kpr591 < 1U; i_kpr591++) {
                uint32_t j_kpr592 = i_kpr591;
                uint32_t start = j_kpr594 * 4U + j_kpr592 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr592 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr595 = 0U;
        for (; i_kpr595 < 2U; i_kpr595++) {
            uint32_t j_kpr596 = i_kpr595;
            vec_memcpy(elems_tile0 + (j_kpr596 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr596 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr597 = 0U;
        for (; i_kpr597 < 2U; i_kpr597++) {
            uint32_t j_kpr598 = i_kpr597;
            vec_memcpy(col_ind_tile0 + (j_kpr598 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr598 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr599 = 0U;
        for (; i_kpr599 < to_; i_kpr599++)
            elems_tile0[i_kpr599 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr601 = 0U;
            for (; i_kpr601 < 2U; i_kpr601++) {
                uint32_t j_kpr602 = i_kpr601;
                vec_memcpy(elems_tile0 + (j_kpr602 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr602 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr603 = 0U;
            for (; i_kpr603 < 2U; i_kpr603++) {
                uint32_t j_kpr604 = i_kpr603;
                vec_memcpy(col_ind_tile0 + (j_kpr604 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr604 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr606 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr605 = 0U;
    for (; i_kpr605 < hi_kpr606; i_kpr605++) {
        uint32_t j_kpr607 = i_kpr605;
        elems_tile0[j_kpr607 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr607 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr607 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr607 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr610 = 0U;
    for (; i_kpr610 < 2U; i_kpr610++) {
        uint32_t j_kpr611 = i_kpr610;
        uint32_t off = n_idx + j_kpr611 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr608 = 0U;
            for (; i_kpr608 < 1U; i_kpr608++) {
                uint32_t j_kpr609 = i_kpr608;
                uint32_t start = j_kpr611 * 4U + j_kpr609 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr609 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr612 = 0U;
        for (; i_kpr612 < 1U; i_kpr612++) {
            uint32_t j_kpr613 = i_kpr612;
            vec_memcpy(elems_tile0 + (j_kpr613 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr613 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr614 = 0U;
        for (; i_kpr614 < 1U; i_kpr614++) {
            uint32_t j_kpr615 = i_kpr614;
            vec_memcpy(col_ind_tile0 + (j_kpr615 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr615 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr616 = 0U;
        for (; i_kpr616 < to_; i_kpr616++)
            elems_tile0[i_kpr616 * 32U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr618 = 0U;
            for (; i_kpr618 < 1U; i_kpr618++) {
                uint32_t j_kpr619 = i_kpr618;
                vec_memcpy(elems_tile0 + (j_kpr619 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr619 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr620 = 0U;
            for (; i_kpr620 < 1U; i_kpr620++) {
                uint32_t j_kpr621 = i_kpr620;
                vec_memcpy(col_ind_tile0 + (j_kpr621 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr621 * 32U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr623 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr622 = 0U;
    for (; i_kpr622 < hi_kpr623; i_kpr622++) {
        uint32_t j_kpr624 = i_kpr622;
        elems_tile0[j_kpr624 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr624 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr624 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr624 * 32U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr627 = 0U;
    for (; i_kpr627 < 1U; i_kpr627++) {
        uint32_t j_kpr628 = i_kpr627;
        uint32_t off = n_idx + j_kpr628 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr625 = 0U;
            for (; i_kpr625 < 1U; i_kpr625++) {
                uint32_t j_kpr626 = i_kpr625;
                uint32_t start = j_kpr628 * 4U + j_kpr626 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr626 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr629 = 0U;
        for (; i_kpr629 < 1U; i_kpr629++) {
            uint32_t j_kpr630 = i_kpr629;
            vec_memcpy(elems_tile0 + (j_kpr630 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr630 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr631 = 0U;
        for (; i_kpr631 < 1U; i_kpr631++) {
            uint32_t j_kpr632 = i_kpr631;
            vec_memcpy(col_ind_tile0 + (j_kpr632 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr632 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr633 = 0U;
        for (; i_kpr633 < to_; i_kpr633++)
            elems_tile0[i_kpr633 * 32U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr635 = 0U;
            for (; i_kpr635 < 1U; i_kpr635++) {
                uint32_t j_kpr636 = i_kpr635;
                vec_memcpy(elems_tile0 + (j_kpr636 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr636 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr637 = 0U;
            for (; i_kpr637 < 1U; i_kpr637++) {
                uint32_t j_kpr638 = i_kpr637;
                vec_memcpy(col_ind_tile0 + (j_kpr638 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr638 * 32U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr640 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr639 = 0U;
    for (; i_kpr639 < hi_kpr640; i_kpr639++) {
        uint32_t j_kpr641 = i_kpr639;
        elems_tile0[j_kpr641 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr641 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr641 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr641 * 32U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr644 = 0U;
    for (; i_kpr644 < 1U; i_kpr644++) {
        uint32_t j_kpr645 = i_kpr644;
        uint32_t off = n_idx + j_kpr645 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr642 = 0U;
            for (; i_kpr642 < 1U; i_kpr642++) {
                uint32_t j_kpr643 = i_kpr642;
                uint32_t start = j_kpr645 * 4U + j_kpr643 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr643 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr646 = 0U;
        for (; i_kpr646 < 2U; i_kpr646++) {
            uint32_t j_kpr647 = i_kpr646;
            vec_memcpy(elems_tile0 + (j_kpr647 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr647 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr648 = 0U;
        for (; i_kpr648 < 2U; i_kpr648++) {
            uint32_t j_kpr649 = i_kpr648;
            vec_memcpy(col_ind_tile0 + (j_kpr649 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr649 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr650 = 0U;
        for (; i_kpr650 < to_; i_kpr650++)
            elems_tile0[i_kpr650 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr652 = 0U;
            for (; i_kpr652 < 2U; i_kpr652++) {
                uint32_t j_kpr653 = i_kpr652;
                vec_memcpy(elems_tile0 + (j_kpr653 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr653 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr654 = 0U;
            for (; i_kpr654 < 2U; i_kpr654++) {
                uint32_t j_kpr655 = i_kpr654;
                vec_memcpy(col_ind_tile0 + (j_kpr655 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr655 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr657 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr656 = 0U;
    for (; i_kpr656 < hi_kpr657; i_kpr656++) {
        uint32_t j_kpr658 = i_kpr656;
        elems_tile0[j_kpr658 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr658 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr658 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr658 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr661 = 0U;
    for (; i_kpr661 < 4U; i_kpr661++) {
        uint32_t j_kpr662 = i_kpr661;
        uint32_t off = n_idx + j_kpr662 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr659 = 0U;
            for (; i_kpr659 < 1U; i_kpr659++) {
                uint32_t j_kpr660 = i_kpr659;
                uint32_t start = j_kpr662 * 4U + j_kpr660 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr660 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr663 = 0U;
        for (; i_kpr663 < 2U; i_kpr663++) {
            uint32_t j_kpr664 = i_kpr663;
            vec_memcpy(elems_tile0 + (j_kpr664 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr664 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr665 = 0U;
        for (; i_kpr665 < 2U; i_kpr665++) {
            uint32_t j_kpr666 = i_kpr665;
            vec_memcpy(col_ind_tile0 + (j_kpr666 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr666 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr667 = 0U;
        for (; i_kpr667 < to_; i_kpr667++)
            elems_tile0[i_kpr667 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr669 = 0U;
            for (; i_kpr669 < 2U; i_kpr669++) {
                uint32_t j_kpr670 = i_kpr669;
                vec_memcpy(elems_tile0 + (j_kpr670 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr670 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr671 = 0U;
            for (; i_kpr671 < 2U; i_kpr671++) {
                uint32_t j_kpr672 = i_kpr671;
                vec_memcpy(col_ind_tile0 + (j_kpr672 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr672 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr674 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr673 = 0U;
    for (; i_kpr673 < hi_kpr674; i_kpr673++) {
        uint32_t j_kpr675 = i_kpr673;
        elems_tile0[j_kpr675 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr675 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr675 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr675 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr678 = 0U;
    for (; i_kpr678 < 4U; i_kpr678++) {
        uint32_t j_kpr679 = i_kpr678;
        uint32_t off = n_idx + j_kpr679 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr676 = 0U;
            for (; i_kpr676 < 1U; i_kpr676++) {
                uint32_t j_kpr677 = i_kpr676;
                uint32_t start = j_kpr679 * 4U + j_kpr677 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr677 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr680 = 0U;
        for (; i_kpr680 < 1U; i_kpr680++) {
            uint32_t j_kpr681 = i_kpr680;
            vec_memcpy(elems_tile0 + (j_kpr681 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr681 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr682 = 0U;
        for (; i_kpr682 < 1U; i_kpr682++) {
            uint32_t j_kpr683 = i_kpr682;
            vec_memcpy(col_ind_tile0 + (j_kpr683 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr683 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr684 = 0U;
        for (; i_kpr684 < to_; i_kpr684++)
            elems_tile0[i_kpr684 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr686 = 0U;
            for (; i_kpr686 < 1U; i_kpr686++) {
                uint32_t j_kpr687 = i_kpr686;
                vec_memcpy(elems_tile0 + (j_kpr687 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr687 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr688 = 0U;
            for (; i_kpr688 < 1U; i_kpr688++) {
                uint32_t j_kpr689 = i_kpr688;
                vec_memcpy(col_ind_tile0 + (j_kpr689 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr689 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr691 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr690 = 0U;
    for (; i_kpr690 < hi_kpr691; i_kpr690++) {
        uint32_t j_kpr692 = i_kpr690;
        elems_tile0[j_kpr692 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr692 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr692 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr692 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr695 = 0U;
    for (; i_kpr695 < 2U; i_kpr695++) {
        uint32_t j_kpr696 = i_kpr695;
        uint32_t off = n_idx + j_kpr696 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr693 = 0U;
            for (; i_kpr693 < 1U; i_kpr693++) {
                uint32_t j_kpr694 = i_kpr693;
                uint32_t start = j_kpr696 * 4U + j_kpr694 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr694 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr697 = 0U;
        for (; i_kpr697 < 1U; i_kpr697++) {
            uint32_t j_kpr698 = i_kpr697;
            vec_memcpy(elems_tile0 + (j_kpr698 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr698 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr699 = 0U;
        for (; i_kpr699 < 1U; i_kpr699++) {
            uint32_t j_kpr700 = i_kpr699;
            vec_memcpy(col_ind_tile0 + (j_kpr700 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr700 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr701 = 0U;
        for (; i_kpr701 < to_; i_kpr701++)
            elems_tile0[i_kpr701 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr703 = 0U;
            for (; i_kpr703 < 1U; i_kpr703++) {
                uint32_t j_kpr704 = i_kpr703;
                vec_memcpy(elems_tile0 + (j_kpr704 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr704 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr705 = 0U;
            for (; i_kpr705 < 1U; i_kpr705++) {
                uint32_t j_kpr706 = i_kpr705;
                vec_memcpy(col_ind_tile0 + (j_kpr706 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr706 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr708 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr707 = 0U;
    for (; i_kpr707 < hi_kpr708; i_kpr707++) {
        uint32_t j_kpr709 = i_kpr707;
        elems_tile0[j_kpr709 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr709 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr709 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr709 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr712 = 0U;
    for (; i_kpr712 < 2U; i_kpr712++) {
        uint32_t j_kpr713 = i_kpr712;
        uint32_t off = n_idx + j_kpr713 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr710 = 0U;
            for (; i_kpr710 < 1U; i_kpr710++) {
                uint32_t j_kpr711 = i_kpr710;
                uint32_t start = j_kpr713 * 4U + j_kpr711 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr711 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr714 = 0U;
        for (; i_kpr714 < 2U; i_kpr714++) {
            uint32_t j_kpr715 = i_kpr714;
            vec_memcpy(elems_tile0 + (j_kpr715 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr715 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr716 = 0U;
        for (; i_kpr716 < 2U; i_kpr716++) {
            uint32_t j_kpr717 = i_kpr716;
            vec_memcpy(col_ind_tile0 + (j_kpr717 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr717 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr718 = 0U;
        for (; i_kpr718 < to_; i_kpr718++)
            elems_tile0[i_kpr718 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr720 = 0U;
            for (; i_kpr720 < 2U; i_kpr720++) {
                uint32_t j_kpr721 = i_kpr720;
                vec_memcpy(elems_tile0 + (j_kpr721 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr721 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr722 = 0U;
            for (; i_kpr722 < 2U; i_kpr722++) {
                uint32_t j_kpr723 = i_kpr722;
                vec_memcpy(col_ind_tile0 + (j_kpr723 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr723 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr725 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr724 = 0U;
    for (; i_kpr724 < hi_kpr725; i_kpr724++) {
        uint32_t j_kpr726 = i_kpr724;
        elems_tile0[j_kpr726 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr726 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr726 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr726 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr729 = 0U;
    for (; i_kpr729 < 8U; i_kpr729++) {
        uint32_t j_kpr730 = i_kpr729;
        uint32_t off = n_idx + j_kpr730 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr727 = 0U;
            for (; i_kpr727 < 1U; i_kpr727++) {
                uint32_t j_kpr728 = i_kpr727;
                uint32_t start = j_kpr730 * 4U + j_kpr728 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr728 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr731 = 0U;
        for (; i_kpr731 < 2U; i_kpr731++) {
            uint32_t j_kpr732 = i_kpr731;
            vec_memcpy(elems_tile0 + (j_kpr732 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr732 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr733 = 0U;
        for (; i_kpr733 < 2U; i_kpr733++) {
            uint32_t j_kpr734 = i_kpr733;
            vec_memcpy(col_ind_tile0 + (j_kpr734 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr734 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr735 = 0U;
        for (; i_kpr735 < to_; i_kpr735++)
            elems_tile0[i_kpr735 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr737 = 0U;
            for (; i_kpr737 < 2U; i_kpr737++) {
                uint32_t j_kpr738 = i_kpr737;
                vec_memcpy(elems_tile0 + (j_kpr738 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr738 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr739 = 0U;
            for (; i_kpr739 < 2U; i_kpr739++) {
                uint32_t j_kpr740 = i_kpr739;
                vec_memcpy(col_ind_tile0 + (j_kpr740 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr740 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr742 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr741 = 0U;
    for (; i_kpr741 < hi_kpr742; i_kpr741++) {
        uint32_t j_kpr743 = i_kpr741;
        elems_tile0[j_kpr743 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr743 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr743 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr743 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr746 = 0U;
    for (; i_kpr746 < 8U; i_kpr746++) {
        uint32_t j_kpr747 = i_kpr746;
        uint32_t off = n_idx + j_kpr747 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr744 = 0U;
            for (; i_kpr744 < 1U; i_kpr744++) {
                uint32_t j_kpr745 = i_kpr744;
                uint32_t start = j_kpr747 * 4U + j_kpr745 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr745 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr748 = 0U;
        for (; i_kpr748 < 1U; i_kpr748++) {
            uint32_t j_kpr749 = i_kpr748;
            vec_memcpy(elems_tile0 + (j_kpr749 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr749 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr750 = 0U;
        for (; i_kpr750 < 1U; i_kpr750++) {
            uint32_t j_kpr751 = i_kpr750;
            vec_memcpy(col_ind_tile0 + (j_kpr751 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr751 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr752 = 0U;
        for (; i_kpr752 < to_; i_kpr752++)
            elems_tile0[i_kpr752 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr754 = 0U;
            for (; i_kpr754 < 1U; i_kpr754++) {
                uint32_t j_kpr755 = i_kpr754;
                vec_memcpy(elems_tile0 + (j_kpr755 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr755 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr756 = 0U;
            for (; i_kpr756 < 1U; i_kpr756++) {
                uint32_t j_kpr757 = i_kpr756;
                vec_memcpy(col_ind_tile0 + (j_kpr757 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr757 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr759 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr758 = 0U;
    for (; i_kpr758 < hi_kpr759; i_kpr758++) {
        uint32_t j_kpr760 = i_kpr758;
        elems_tile0[j_kpr760 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr760 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr760 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr760 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr763 = 0U;
    for (; i_kpr763 < 4U; i_kpr763++) {
        uint32_t j_kpr764 = i_kpr763;
        uint32_t off = n_idx + j_kpr764 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr761 = 0U;
            for (; i_kpr761 < 1U; i_kpr761++) {
                uint32_t j_kpr762 = i_kpr761;
                uint32_t start = j_kpr764 * 4U + j_kpr762 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr762 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr765 = 0U;
        for (; i_kpr765 < 1U; i_kpr765++) {
            uint32_t j_kpr766 = i_kpr765;
            vec_memcpy(elems_tile0 + (j_kpr766 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr766 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr767 = 0U;
        for (; i_kpr767 < 1U; i_kpr767++) {
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr771 = 0U;
            for (; i_kpr771 < 1U; i_kpr771++) {
                uint32_t j_kpr772 = i_kpr771;
                vec_memcpy(elems_tile0 + (j_kpr772 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr772 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr773 = 0U;
            for (; i_kpr773 < 1U; i_kpr773++) {
                uint32_t j_kpr774 = i_kpr773;
                vec_memcpy(col_ind_tile0 + (j_kpr774 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr774 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr780 = 0U;
    for (; i_kpr780 < 4U; i_kpr780++) {
        uint32_t j_kpr781 = i_kpr780;
        uint32_t off = n_idx + j_kpr781 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr778 = 0U;
            for (; i_kpr778 < 1U; i_kpr778++) {
                uint32_t j_kpr779 = i_kpr778;
                uint32_t start = j_kpr781 * 4U + j_kpr779 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr779 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr782 = 0U;
        for (; i_kpr782 < 4U; i_kpr782++) {
            uint32_t j_kpr783 = i_kpr782;
            vec_memcpy(elems_tile0 + (j_kpr783 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr783 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr784 = 0U;
        for (; i_kpr784 < 4U; i_kpr784++) {
            uint32_t j_kpr785 = i_kpr784;
            vec_memcpy(col_ind_tile0 + (j_kpr785 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr785 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr786 = 0U;
        for (; i_kpr786 < to_; i_kpr786++)
            elems_tile0[i_kpr786 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr788 = 0U;
            for (; i_kpr788 < 4U; i_kpr788++) {
                uint32_t j_kpr789 = i_kpr788;
                vec_memcpy(elems_tile0 + (j_kpr789 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr789 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr790 = 0U;
            for (; i_kpr790 < 4U; i_kpr790++) {
                uint32_t j_kpr791 = i_kpr790;
                vec_memcpy(col_ind_tile0 + (j_kpr791 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr791 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr793 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr792 = 0U;
    for (; i_kpr792 < hi_kpr793; i_kpr792++) {
        uint32_t j_kpr794 = i_kpr792;
        elems_tile0[j_kpr794 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr794 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr794 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr794 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr797 = 0U;
    for (; i_kpr797 < 1U; i_kpr797++) {
        uint32_t j_kpr798 = i_kpr797;
        uint32_t off = n_idx + j_kpr798 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr795 = 0U;
            for (; i_kpr795 < 1U; i_kpr795++) {
                uint32_t j_kpr796 = i_kpr795;
                uint32_t start = j_kpr798 * 4U + j_kpr796 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr796 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr799 = 0U;
        for (; i_kpr799 < 4U; i_kpr799++) {
            uint32_t j_kpr800 = i_kpr799;
            vec_memcpy(elems_tile0 + (j_kpr800 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr800 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr801 = 0U;
        for (; i_kpr801 < 4U; i_kpr801++) {
            uint32_t j_kpr802 = i_kpr801;
            vec_memcpy(col_ind_tile0 + (j_kpr802 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr802 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr803 = 0U;
        for (; i_kpr803 < to_; i_kpr803++)
            elems_tile0[i_kpr803 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr805 = 0U;
            for (; i_kpr805 < 4U; i_kpr805++) {
                uint32_t j_kpr806 = i_kpr805;
                vec_memcpy(elems_tile0 + (j_kpr806 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr806 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr807 = 0U;
            for (; i_kpr807 < 4U; i_kpr807++) {
                uint32_t j_kpr808 = i_kpr807;
                vec_memcpy(col_ind_tile0 + (j_kpr808 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr808 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr810 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr809 = 0U;
    for (; i_kpr809 < hi_kpr810; i_kpr809++) {
        uint32_t j_kpr811 = i_kpr809;
        elems_tile0[j_kpr811 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr811 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr811 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr811 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr814 = 0U;
    for (; i_kpr814 < 1U; i_kpr814++) {
        uint32_t j_kpr815 = i_kpr814;
        uint32_t off = n_idx + j_kpr815 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr812 = 0U;
            for (; i_kpr812 < 1U; i_kpr812++) {
                uint32_t j_kpr813 = i_kpr812;
                uint32_t start = j_kpr815 * 4U + j_kpr813 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr813 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr816 = 0U;
        for (; i_kpr816 < 4U; i_kpr816++) {
            uint32_t j_kpr817 = i_kpr816;
            vec_memcpy(elems_tile0 + (j_kpr817 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr817 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr818 = 0U;
        for (; i_kpr818 < 4U; i_kpr818++) {
            uint32_t j_kpr819 = i_kpr818;
            vec_memcpy(col_ind_tile0 + (j_kpr819 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr819 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr820 = 0U;
        for (; i_kpr820 < to_; i_kpr820++)
            elems_tile0[i_kpr820 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr822 = 0U;
            for (; i_kpr822 < 4U; i_kpr822++) {
                uint32_t j_kpr823 = i_kpr822;
                vec_memcpy(elems_tile0 + (j_kpr823 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr823 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr824 = 0U;
            for (; i_kpr824 < 4U; i_kpr824++) {
                uint32_t j_kpr825 = i_kpr824;
                vec_memcpy(col_ind_tile0 + (j_kpr825 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr825 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr827 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr826 = 0U;
    for (; i_kpr826 < hi_kpr827; i_kpr826++) {
        uint32_t j_kpr828 = i_kpr826;
        elems_tile0[j_kpr828 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr828 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr828 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr828 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr831 = 0U;
    for (; i_kpr831 < 2U; i_kpr831++) {
        uint32_t j_kpr832 = i_kpr831;
        uint32_t off = n_idx + j_kpr832 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr829 = 0U;
            for (; i_kpr829 < 1U; i_kpr829++) {
                uint32_t j_kpr830 = i_kpr829;
                uint32_t start = j_kpr832 * 4U + j_kpr830 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr830 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr833 = 0U;
        for (; i_kpr833 < 4U; i_kpr833++) {
            uint32_t j_kpr834 = i_kpr833;
            vec_memcpy(elems_tile0 + (j_kpr834 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr834 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr835 = 0U;
        for (; i_kpr835 < 4U; i_kpr835++) {
            uint32_t j_kpr836 = i_kpr835;
            vec_memcpy(col_ind_tile0 + (j_kpr836 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr836 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr837 = 0U;
        for (; i_kpr837 < to_; i_kpr837++)
            elems_tile0[i_kpr837 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr839 = 0U;
            for (; i_kpr839 < 4U; i_kpr839++) {
                uint32_t j_kpr840 = i_kpr839;
                vec_memcpy(elems_tile0 + (j_kpr840 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr840 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr841 = 0U;
            for (; i_kpr841 < 4U; i_kpr841++) {
                uint32_t j_kpr842 = i_kpr841;
                vec_memcpy(col_ind_tile0 + (j_kpr842 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr842 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr844 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr843 = 0U;
    for (; i_kpr843 < hi_kpr844; i_kpr843++) {
        uint32_t j_kpr845 = i_kpr843;
        elems_tile0[j_kpr845 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr845 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr845 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr845 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr848 = 0U;
    for (; i_kpr848 < 2U; i_kpr848++) {
        uint32_t j_kpr849 = i_kpr848;
        uint32_t off = n_idx + j_kpr849 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr846 = 0U;
            for (; i_kpr846 < 1U; i_kpr846++) {
                uint32_t j_kpr847 = i_kpr846;
                uint32_t start = j_kpr849 * 4U + j_kpr847 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr847 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr850 = 0U;
        for (; i_kpr850 < 2U; i_kpr850++) {
            uint32_t j_kpr851 = i_kpr850;
            vec_memcpy(elems_tile0 + (j_kpr851 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr851 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr852 = 0U;
        for (; i_kpr852 < 2U; i_kpr852++) {
            uint32_t j_kpr853 = i_kpr852;
            vec_memcpy(col_ind_tile0 + (j_kpr853 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr853 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr854 = 0U;
        for (; i_kpr854 < to_; i_kpr854++)
            elems_tile0[i_kpr854 * 32U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr856 = 0U;
            for (; i_kpr856 < 2U; i_kpr856++) {
                uint32_t j_kpr857 = i_kpr856;
                vec_memcpy(elems_tile0 + (j_kpr857 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr857 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr858 = 0U;
            for (; i_kpr858 < 2U; i_kpr858++) {
                uint32_t j_kpr859 = i_kpr858;
                vec_memcpy(col_ind_tile0 + (j_kpr859 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr859 * 32U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr861 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr860 = 0U;
    for (; i_kpr860 < hi_kpr861; i_kpr860++) {
        uint32_t j_kpr862 = i_kpr860;
        elems_tile0[j_kpr862 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr862 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr862 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr862 * 32U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr865 = 0U;
    for (; i_kpr865 < 1U; i_kpr865++) {
        uint32_t j_kpr866 = i_kpr865;
        uint32_t off = n_idx + j_kpr866 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr863 = 0U;
            for (; i_kpr863 < 1U; i_kpr863++) {
                uint32_t j_kpr864 = i_kpr863;
                uint32_t start = j_kpr866 * 4U + j_kpr864 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr864 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr867 = 0U;
        for (; i_kpr867 < 2U; i_kpr867++) {
            uint32_t j_kpr868 = i_kpr867;
            vec_memcpy(elems_tile0 + (j_kpr868 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr868 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr869 = 0U;
        for (; i_kpr869 < 2U; i_kpr869++) {
            uint32_t j_kpr870 = i_kpr869;
            vec_memcpy(col_ind_tile0 + (j_kpr870 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr870 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr871 = 0U;
        for (; i_kpr871 < to_; i_kpr871++)
            elems_tile0[i_kpr871 * 32U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr873 = 0U;
            for (; i_kpr873 < 2U; i_kpr873++) {
                uint32_t j_kpr874 = i_kpr873;
                vec_memcpy(elems_tile0 + (j_kpr874 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr874 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr875 = 0U;
            for (; i_kpr875 < 2U; i_kpr875++) {
                uint32_t j_kpr876 = i_kpr875;
                vec_memcpy(col_ind_tile0 + (j_kpr876 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr876 * 32U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr878 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr877 = 0U;
    for (; i_kpr877 < hi_kpr878; i_kpr877++) {
        uint32_t j_kpr879 = i_kpr877;
        elems_tile0[j_kpr879 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr879 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr879 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr879 * 32U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr882 = 0U;
    for (; i_kpr882 < 1U; i_kpr882++) {
        uint32_t j_kpr883 = i_kpr882;
        uint32_t off = n_idx + j_kpr883 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr880 = 0U;
            for (; i_kpr880 < 1U; i_kpr880++) {
                uint32_t j_kpr881 = i_kpr880;
                uint32_t start = j_kpr883 * 4U + j_kpr881 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr881 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr884 = 0U;
        for (; i_kpr884 < 4U; i_kpr884++) {
            uint32_t j_kpr885 = i_kpr884;
            vec_memcpy(elems_tile0 + (j_kpr885 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr885 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr886 = 0U;
        for (; i_kpr886 < 4U; i_kpr886++) {
            uint32_t j_kpr887 = i_kpr886;
            vec_memcpy(col_ind_tile0 + (j_kpr887 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr887 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr888 = 0U;
        for (; i_kpr888 < to_; i_kpr888++)
            elems_tile0[i_kpr888 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr890 = 0U;
            for (; i_kpr890 < 4U; i_kpr890++) {
                uint32_t j_kpr891 = i_kpr890;
                vec_memcpy(elems_tile0 + (j_kpr891 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr891 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr892 = 0U;
            for (; i_kpr892 < 4U; i_kpr892++) {
                uint32_t j_kpr893 = i_kpr892;
                vec_memcpy(col_ind_tile0 + (j_kpr893 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr893 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr895 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr894 = 0U;
    for (; i_kpr894 < hi_kpr895; i_kpr894++) {
        uint32_t j_kpr896 = i_kpr894;
        elems_tile0[j_kpr896 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr896 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr896 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr896 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr899 = 0U;
    for (; i_kpr899 < 4U; i_kpr899++) {
        uint32_t j_kpr900 = i_kpr899;
        uint32_t off = n_idx + j_kpr900 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr897 = 0U;
            for (; i_kpr897 < 1U; i_kpr897++) {
                uint32_t j_kpr898 = i_kpr897;
                uint32_t start = j_kpr900 * 4U + j_kpr898 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr898 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr901 = 0U;
        for (; i_kpr901 < 4U; i_kpr901++) {
            uint32_t j_kpr902 = i_kpr901;
            vec_memcpy(elems_tile0 + (j_kpr902 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr902 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr903 = 0U;
        for (; i_kpr903 < 4U; i_kpr903++) {
            uint32_t j_kpr904 = i_kpr903;
            vec_memcpy(col_ind_tile0 + (j_kpr904 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr904 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr905 = 0U;
        for (; i_kpr905 < to_; i_kpr905++)
            elems_tile0[i_kpr905 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr907 = 0U;
            for (; i_kpr907 < 4U; i_kpr907++) {
                uint32_t j_kpr908 = i_kpr907;
                vec_memcpy(elems_tile0 + (j_kpr908 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr908 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr909 = 0U;
            for (; i_kpr909 < 4U; i_kpr909++) {
                uint32_t j_kpr910 = i_kpr909;
                vec_memcpy(col_ind_tile0 + (j_kpr910 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr910 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr912 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr911 = 0U;
    for (; i_kpr911 < hi_kpr912; i_kpr911++) {
        uint32_t j_kpr913 = i_kpr911;
        elems_tile0[j_kpr913 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr913 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr913 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr913 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr916 = 0U;
    for (; i_kpr916 < 4U; i_kpr916++) {
        uint32_t j_kpr917 = i_kpr916;
        uint32_t off = n_idx + j_kpr917 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr914 = 0U;
            for (; i_kpr914 < 1U; i_kpr914++) {
                uint32_t j_kpr915 = i_kpr914;
                uint32_t start = j_kpr917 * 4U + j_kpr915 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr915 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr918 = 0U;
        for (; i_kpr918 < 2U; i_kpr918++) {
            uint32_t j_kpr919 = i_kpr918;
            vec_memcpy(elems_tile0 + (j_kpr919 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr919 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr920 = 0U;
        for (; i_kpr920 < 2U; i_kpr920++) {
            uint32_t j_kpr921 = i_kpr920;
            vec_memcpy(col_ind_tile0 + (j_kpr921 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr921 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr922 = 0U;
        for (; i_kpr922 < to_; i_kpr922++)
            elems_tile0[i_kpr922 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr924 = 0U;
            for (; i_kpr924 < 2U; i_kpr924++) {
                uint32_t j_kpr925 = i_kpr924;
                vec_memcpy(elems_tile0 + (j_kpr925 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr925 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr926 = 0U;
            for (; i_kpr926 < 2U; i_kpr926++) {
                uint32_t j_kpr927 = i_kpr926;
                vec_memcpy(col_ind_tile0 + (j_kpr927 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr927 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr929 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr928 = 0U;
    for (; i_kpr928 < hi_kpr929; i_kpr928++) {
        uint32_t j_kpr930 = i_kpr928;
        elems_tile0[j_kpr930 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr930 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr930 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr930 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr933 = 0U;
    for (; i_kpr933 < 2U; i_kpr933++) {
        uint32_t j_kpr934 = i_kpr933;
        uint32_t off = n_idx + j_kpr934 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr931 = 0U;
            for (; i_kpr931 < 1U; i_kpr931++) {
                uint32_t j_kpr932 = i_kpr931;
                uint32_t start = j_kpr934 * 4U + j_kpr932 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr932 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr935 = 0U;
        for (; i_kpr935 < 2U; i_kpr935++) {
            uint32_t j_kpr936 = i_kpr935;
            vec_memcpy(elems_tile0 + (j_kpr936 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr936 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr937 = 0U;
        for (; i_kpr937 < 2U; i_kpr937++) {
            uint32_t j_kpr938 = i_kpr937;
            vec_memcpy(col_ind_tile0 + (j_kpr938 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr938 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr939 = 0U;
        for (; i_kpr939 < to_; i_kpr939++)
            elems_tile0[i_kpr939 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr941 = 0U;
            for (; i_kpr941 < 2U; i_kpr941++) {
                uint32_t j_kpr942 = i_kpr941;
                vec_memcpy(elems_tile0 + (j_kpr942 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr942 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr943 = 0U;
            for (; i_kpr943 < 2U; i_kpr943++) {
                uint32_t j_kpr944 = i_kpr943;
                vec_memcpy(col_ind_tile0 + (j_kpr944 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr944 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr946 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr945 = 0U;
    for (; i_kpr945 < hi_kpr946; i_kpr945++) {
        uint32_t j_kpr947 = i_kpr945;
        elems_tile0[j_kpr947 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr947 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr947 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr947 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr950 = 0U;
    for (; i_kpr950 < 2U; i_kpr950++) {
        uint32_t j_kpr951 = i_kpr950;
        uint32_t off = n_idx + j_kpr951 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr948 = 0U;
            for (; i_kpr948 < 1U; i_kpr948++) {
                uint32_t j_kpr949 = i_kpr948;
                uint32_t start = j_kpr951 * 4U + j_kpr949 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr949 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr952 = 0U;
        for (; i_kpr952 < 1U; i_kpr952++) {
            uint32_t j_kpr953 = i_kpr952;
            vec_memcpy(elems_tile0 + (j_kpr953 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr953 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr954 = 0U;
        for (; i_kpr954 < 1U; i_kpr954++) {
            uint32_t j_kpr955 = i_kpr954;
            vec_memcpy(col_ind_tile0 + (j_kpr955 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr955 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr956 = 0U;
        for (; i_kpr956 < to_; i_kpr956++)
            elems_tile0[i_kpr956 * 64U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr958 = 0U;
            for (; i_kpr958 < 1U; i_kpr958++) {
                uint32_t j_kpr959 = i_kpr958;
                vec_memcpy(elems_tile0 + (j_kpr959 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr959 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr960 = 0U;
            for (; i_kpr960 < 1U; i_kpr960++) {
                uint32_t j_kpr961 = i_kpr960;
                vec_memcpy(col_ind_tile0 + (j_kpr961 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr961 * 64U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr963 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr962 = 0U;
    for (; i_kpr962 < hi_kpr963; i_kpr962++) {
        uint32_t j_kpr964 = i_kpr962;
        elems_tile0[j_kpr964 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr964 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr964 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr964 * 64U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr967 = 0U;
    for (; i_kpr967 < 1U; i_kpr967++) {
        uint32_t j_kpr968 = i_kpr967;
        uint32_t off = n_idx + j_kpr968 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr965 = 0U;
            for (; i_kpr965 < 1U; i_kpr965++) {
                uint32_t j_kpr966 = i_kpr965;
                uint32_t start = j_kpr968 * 4U + j_kpr966 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr966 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr969 = 0U;
        for (; i_kpr969 < 1U; i_kpr969++) {
            uint32_t j_kpr970 = i_kpr969;
            vec_memcpy(elems_tile0 + (j_kpr970 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr970 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr971 = 0U;
        for (; i_kpr971 < 1U; i_kpr971++) {
            uint32_t j_kpr972 = i_kpr971;
            vec_memcpy(col_ind_tile0 + (j_kpr972 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr972 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr973 = 0U;
        for (; i_kpr973 < to_; i_kpr973++)
            elems_tile0[i_kpr973 * 64U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr975 = 0U;
            for (; i_kpr975 < 1U; i_kpr975++) {
                uint32_t j_kpr976 = i_kpr975;
                vec_memcpy(elems_tile0 + (j_kpr976 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr976 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr977 = 0U;
            for (; i_kpr977 < 1U; i_kpr977++) {
                uint32_t j_kpr978 = i_kpr977;
                vec_memcpy(col_ind_tile0 + (j_kpr978 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr978 * 64U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr980 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr979 = 0U;
    for (; i_kpr979 < hi_kpr980; i_kpr979++) {
        uint32_t j_kpr981 = i_kpr979;
        elems_tile0[j_kpr981 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr981 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr981 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr981 * 64U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr984 = 0U;
    for (; i_kpr984 < 1U; i_kpr984++) {
        uint32_t j_kpr985 = i_kpr984;
        uint32_t off = n_idx + j_kpr985 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr982 = 0U;
            for (; i_kpr982 < 1U; i_kpr982++) {
                uint32_t j_kpr983 = i_kpr982;
                uint32_t start = j_kpr985 * 4U + j_kpr983 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr983 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr986 = 0U;
        for (; i_kpr986 < 4U; i_kpr986++) {
            uint32_t j_kpr987 = i_kpr986;
            vec_memcpy(elems_tile0 + (j_kpr987 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr987 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr988 = 0U;
        for (; i_kpr988 < 4U; i_kpr988++) {
            uint32_t j_kpr989 = i_kpr988;
            vec_memcpy(col_ind_tile0 + (j_kpr989 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr989 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr990 = 0U;
        for (; i_kpr990 < to_; i_kpr990++)
            elems_tile0[i_kpr990 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr992 = 0U;
            for (; i_kpr992 < 4U; i_kpr992++) {
                uint32_t j_kpr993 = i_kpr992;
                vec_memcpy(elems_tile0 + (j_kpr993 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr993 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr994 = 0U;
            for (; i_kpr994 < 4U; i_kpr994++) {
                uint32_t j_kpr995 = i_kpr994;
                vec_memcpy(col_ind_tile0 + (j_kpr995 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr995 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr997 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr996 = 0U;
    for (; i_kpr996 < hi_kpr997; i_kpr996++) {
        uint32_t j_kpr998 = i_kpr996;
        elems_tile0[j_kpr998 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr998 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr998 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr998 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1001 = 0U;
    for (; i_kpr1001 < 8U; i_kpr1001++) {
        uint32_t j_kpr1002 = i_kpr1001;
        uint32_t off = n_idx + j_kpr1002 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr999 = 0U;
            for (; i_kpr999 < 1U; i_kpr999++) {
                uint32_t j_kpr1000 = i_kpr999;
                uint32_t start = j_kpr1002 * 4U + j_kpr1000 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1000 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1003 = 0U;
        for (; i_kpr1003 < 4U; i_kpr1003++) {
            uint32_t j_kpr1004 = i_kpr1003;
            vec_memcpy(elems_tile0 + (j_kpr1004 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1004 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1005 = 0U;
        for (; i_kpr1005 < 4U; i_kpr1005++) {
            uint32_t j_kpr1006 = i_kpr1005;
            vec_memcpy(col_ind_tile0 + (j_kpr1006 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1006 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1007 = 0U;
        for (; i_kpr1007 < to_; i_kpr1007++)
            elems_tile0[i_kpr1007 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1009 = 0U;
            for (; i_kpr1009 < 4U; i_kpr1009++) {
                uint32_t j_kpr1010 = i_kpr1009;
                vec_memcpy(elems_tile0 + (j_kpr1010 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1010 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1011 = 0U;
            for (; i_kpr1011 < 4U; i_kpr1011++) {
                uint32_t j_kpr1012 = i_kpr1011;
                vec_memcpy(col_ind_tile0 + (j_kpr1012 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1012 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1014 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1013 = 0U;
    for (; i_kpr1013 < hi_kpr1014; i_kpr1013++) {
        uint32_t j_kpr1015 = i_kpr1013;
        elems_tile0[j_kpr1015 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1015 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1015 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1015 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1018 = 0U;
    for (; i_kpr1018 < 8U; i_kpr1018++) {
        uint32_t j_kpr1019 = i_kpr1018;
        uint32_t off = n_idx + j_kpr1019 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1016 = 0U;
            for (; i_kpr1016 < 1U; i_kpr1016++) {
                uint32_t j_kpr1017 = i_kpr1016;
                uint32_t start = j_kpr1019 * 4U + j_kpr1017 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1017 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1020 = 0U;
        for (; i_kpr1020 < 2U; i_kpr1020++) {
            uint32_t j_kpr1021 = i_kpr1020;
            vec_memcpy(elems_tile0 + (j_kpr1021 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1021 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1022 = 0U;
        for (; i_kpr1022 < 2U; i_kpr1022++) {
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1026 = 0U;
            for (; i_kpr1026 < 2U; i_kpr1026++) {
                uint32_t j_kpr1027 = i_kpr1026;
                vec_memcpy(elems_tile0 + (j_kpr1027 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1027 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1028 = 0U;
            for (; i_kpr1028 < 2U; i_kpr1028++) {
                uint32_t j_kpr1029 = i_kpr1028;
                vec_memcpy(col_ind_tile0 + (j_kpr1029 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1029 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
            for (; k1 < 4U; k1++) {
                uint32_t __anf1 = k1;
                uint32_t __anf02 = k1;
                if (n_idx + __anf02 * 32U * 4U < cols) {
                    float lchunk[4U];
                    memset(lchunk, 0U, 4U * sizeof(float));
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1035 = 0U;
    for (; i_kpr1035 < 4U; i_kpr1035++) {
        uint32_t j_kpr1036 = i_kpr1035;
        uint32_t off = n_idx + j_kpr1036 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1033 = 0U;
            for (; i_kpr1033 < 1U; i_kpr1033++) {
                uint32_t j_kpr1034 = i_kpr1033;
                uint32_t start = j_kpr1036 * 4U + j_kpr1034 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1034 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1037 = 0U;
        for (; i_kpr1037 < 2U; i_kpr1037++) {
            uint32_t j_kpr1038 = i_kpr1037;
            vec_memcpy(elems_tile0 + (j_kpr1038 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1038 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1039 = 0U;
        for (; i_kpr1039 < 2U; i_kpr1039++) {
            uint32_t j_kpr1040 = i_kpr1039;
            vec_memcpy(col_ind_tile0 + (j_kpr1040 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1040 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1041 = 0U;
        for (; i_kpr1041 < to_; i_kpr1041++)
            elems_tile0[i_kpr1041 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1043 = 0U;
            for (; i_kpr1043 < 2U; i_kpr1043++) {
                uint32_t j_kpr1044 = i_kpr1043;
                vec_memcpy(elems_tile0 + (j_kpr1044 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1044 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1045 = 0U;
            for (; i_kpr1045 < 2U; i_kpr1045++) {
                uint32_t j_kpr1046 = i_kpr1045;
                vec_memcpy(col_ind_tile0 + (j_kpr1046 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1046 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1048 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1047 = 0U;
    for (; i_kpr1047 < hi_kpr1048; i_kpr1047++) {
        uint32_t j_kpr1049 = i_kpr1047;
        elems_tile0[j_kpr1049 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1049 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1049 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1049 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1052 = 0U;
    for (; i_kpr1052 < 4U; i_kpr1052++) {
        uint32_t j_kpr1053 = i_kpr1052;
        uint32_t off = n_idx + j_kpr1053 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1050 = 0U;
            for (; i_kpr1050 < 1U; i_kpr1050++) {
                uint32_t j_kpr1051 = i_kpr1050;
                uint32_t start = j_kpr1053 * 4U + j_kpr1051 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1051 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1054 = 0U;
        for (; i_kpr1054 < 1U; i_kpr1054++) {
            uint32_t j_kpr1055 = i_kpr1054;
            vec_memcpy(elems_tile0 + (j_kpr1055 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1055 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1056 = 0U;
        for (; i_kpr1056 < 1U; i_kpr1056++) {
            uint32_t j_kpr1057 = i_kpr1056;
            vec_memcpy(col_ind_tile0 + (j_kpr1057 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1057 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1058 = 0U;
        for (; i_kpr1058 < to_; i_kpr1058++)
            elems_tile0[i_kpr1058 * 64U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1060 = 0U;
            for (; i_kpr1060 < 1U; i_kpr1060++) {
                uint32_t j_kpr1061 = i_kpr1060;
                vec_memcpy(elems_tile0 + (j_kpr1061 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1061 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1062 = 0U;
            for (; i_kpr1062 < 1U; i_kpr1062++) {
                uint32_t j_kpr1063 = i_kpr1062;
                vec_memcpy(col_ind_tile0 + (j_kpr1063 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1063 * 64U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 64U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1065 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1064 = 0U;
    for (; i_kpr1064 < hi_kpr1065; i_kpr1064++) {
        uint32_t j_kpr1066 = i_kpr1064;
        elems_tile0[j_kpr1066 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1066 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1066 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1066 * 64U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1069 = 0U;
    for (; i_kpr1069 < 2U; i_kpr1069++) {
        uint32_t j_kpr1070 = i_kpr1069;
        uint32_t off = n_idx + j_kpr1070 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr1067 = 0U;
            for (; i_kpr1067 < 1U; i_kpr1067++) {
                uint32_t j_kpr1068 = i_kpr1067;
                uint32_t start = j_kpr1070 * 4U + j_kpr1068 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1068 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1071 = 0U;
        for (; i_kpr1071 < 1U; i_kpr1071++) {
            uint32_t j_kpr1072 = i_kpr1071;
            vec_memcpy(elems_tile0 + (j_kpr1072 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1072 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1073 = 0U;
        for (; i_kpr1073 < 1U; i_kpr1073++) {
            uint32_t j_kpr1074 = i_kpr1073;
            vec_memcpy(col_ind_tile0 + (j_kpr1074 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1074 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1075 = 0U;
        for (; i_kpr1075 < to_; i_kpr1075++)
            elems_tile0[i_kpr1075 * 64U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1077 = 0U;
            for (; i_kpr1077 < 1U; i_kpr1077++) {
                uint32_t j_kpr1078 = i_kpr1077;
                vec_memcpy(elems_tile0 + (j_kpr1078 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1078 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1079 = 0U;
            for (; i_kpr1079 < 1U; i_kpr1079++) {
                uint32_t j_kpr1080 = i_kpr1079;
                vec_memcpy(col_ind_tile0 + (j_kpr1080 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1080 * 64U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 64U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1082 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1081 = 0U;
    for (; i_kpr1081 < hi_kpr1082; i_kpr1081++) {
        uint32_t j_kpr1083 = i_kpr1081;
        elems_tile0[j_kpr1083 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1083 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1083 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1083 * 64U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1086 = 0U;
    for (; i_kpr1086 < 2U; i_kpr1086++) {
        uint32_t j_kpr1087 = i_kpr1086;
        uint32_t off = n_idx + j_kpr1087 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr1084 = 0U;
            for (; i_kpr1084 < 1U; i_kpr1084++) {
                uint32_t j_kpr1085 = i_kpr1084;
                uint32_t start = j_kpr1087 * 4U + j_kpr1085 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1085 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1088 = 0U;
        for (; i_kpr1088 < 8U; i_kpr1088++) {
            uint32_t j_kpr1089 = i_kpr1088;
            vec_memcpy(elems_tile0 + (j_kpr1089 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1089 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1090 = 0U;
        for (; i_kpr1090 < 8U; i_kpr1090++) {
            uint32_t j_kpr1091 = i_kpr1090;
            vec_memcpy(col_ind_tile0 + (j_kpr1091 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1091 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1092 = 0U;
        for (; i_kpr1092 < to_; i_kpr1092++)
            elems_tile0[i_kpr1092 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1094 = 0U;
            for (; i_kpr1094 < 8U; i_kpr1094++) {
                uint32_t j_kpr1095 = i_kpr1094;
                vec_memcpy(elems_tile0 + (j_kpr1095 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1095 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1096 = 0U;
            for (; i_kpr1096 < 8U; i_kpr1096++) {
                uint32_t j_kpr1097 = i_kpr1096;
                vec_memcpy(col_ind_tile0 + (j_kpr1097 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1097 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1099 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1098 = 0U;
    for (; i_kpr1098 < hi_kpr1099; i_kpr1098++) {
        uint32_t j_kpr1100 = i_kpr1098;
        elems_tile0[j_kpr1100 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1100 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1100 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1100 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1103 = 0U;
    for (; i_kpr1103 < 1U; i_kpr1103++) {
        uint32_t j_kpr1104 = i_kpr1103;
        uint32_t off = n_idx + j_kpr1104 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1101 = 0U;
            for (; i_kpr1101 < 1U; i_kpr1101++) {
                uint32_t j_kpr1102 = i_kpr1101;
                uint32_t start = j_kpr1104 * 4U + j_kpr1102 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1102 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1105 = 0U;
        for (; i_kpr1105 < 8U; i_kpr1105++) {
            uint32_t j_kpr1106 = i_kpr1105;
            vec_memcpy(elems_tile0 + (j_kpr1106 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1106 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1107 = 0U;
        for (; i_kpr1107 < 8U; i_kpr1107++) {
            uint32_t j_kpr1108 = i_kpr1107;
            vec_memcpy(col_ind_tile0 + (j_kpr1108 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1108 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1109 = 0U;
        for (; i_kpr1109 < to_; i_kpr1109++)
            elems_tile0[i_kpr1109 * 16U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1111 = 0U;
            for (; i_kpr1111 < 8U; i_kpr1111++) {
                uint32_t j_kpr1112 = i_kpr1111;
                vec_memcpy(elems_tile0 + (j_kpr1112 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1112 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1113 = 0U;
            for (; i_kpr1113 < 8U; i_kpr1113++) {
                uint32_t j_kpr1114 = i_kpr1113;
                vec_memcpy(col_ind_tile0 + (j_kpr1114 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1114 * 16U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1116 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1115 = 0U;
    for (; i_kpr1115 < hi_kpr1116; i_kpr1115++) {
        uint32_t j_kpr1117 = i_kpr1115;
        elems_tile0[j_kpr1117 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1117 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1117 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1117 * 16U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1120 = 0U;
    for (; i_kpr1120 < 1U; i_kpr1120++) {
        uint32_t j_kpr1121 = i_kpr1120;
        uint32_t off = n_idx + j_kpr1121 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1118 = 0U;
            for (; i_kpr1118 < 1U; i_kpr1118++) {
                uint32_t j_kpr1119 = i_kpr1118;
                uint32_t start = j_kpr1121 * 4U + j_kpr1119 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1119 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1122 = 0U;
        for (; i_kpr1122 < 8U; i_kpr1122++) {
            uint32_t j_kpr1123 = i_kpr1122;
            vec_memcpy(elems_tile0 + (j_kpr1123 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1123 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1124 = 0U;
        for (; i_kpr1124 < 8U; i_kpr1124++) {
            uint32_t j_kpr1125 = i_kpr1124;
            vec_memcpy(col_ind_tile0 + (j_kpr1125 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1125 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1126 = 0U;
        for (; i_kpr1126 < to_; i_kpr1126++)
            elems_tile0[i_kpr1126 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1128 = 0U;
            for (; i_kpr1128 < 8U; i_kpr1128++) {
                uint32_t j_kpr1129 = i_kpr1128;
                vec_memcpy(elems_tile0 + (j_kpr1129 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1129 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1130 = 0U;
            for (; i_kpr1130 < 8U; i_kpr1130++) {
                uint32_t j_kpr1131 = i_kpr1130;
                vec_memcpy(col_ind_tile0 + (j_kpr1131 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1131 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1133 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1132 = 0U;
    for (; i_kpr1132 < hi_kpr1133; i_kpr1132++) {
        uint32_t j_kpr1134 = i_kpr1132;
        elems_tile0[j_kpr1134 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1134 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1134 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1134 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1137 = 0U;
    for (; i_kpr1137 < 2U; i_kpr1137++) {
        uint32_t j_kpr1138 = i_kpr1137;
        uint32_t off = n_idx + j_kpr1138 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1135 = 0U;
            for (; i_kpr1135 < 1U; i_kpr1135++) {
                uint32_t j_kpr1136 = i_kpr1135;
                uint32_t start = j_kpr1138 * 4U + j_kpr1136 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1136 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1139 = 0U;
        for (; i_kpr1139 < 8U; i_kpr1139++) {
            uint32_t j_kpr1140 = i_kpr1139;
            vec_memcpy(elems_tile0 + (j_kpr1140 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1140 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1141 = 0U;
        for (; i_kpr1141 < 8U; i_kpr1141++) {
            uint32_t j_kpr1142 = i_kpr1141;
            vec_memcpy(col_ind_tile0 + (j_kpr1142 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1142 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1143 = 0U;
        for (; i_kpr1143 < to_; i_kpr1143++)
            elems_tile0[i_kpr1143 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1145 = 0U;
            for (; i_kpr1145 < 8U; i_kpr1145++) {
                uint32_t j_kpr1146 = i_kpr1145;
                vec_memcpy(elems_tile0 + (j_kpr1146 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1146 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1147 = 0U;
            for (; i_kpr1147 < 8U; i_kpr1147++) {
                uint32_t j_kpr1148 = i_kpr1147;
                vec_memcpy(col_ind_tile0 + (j_kpr1148 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1148 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1150 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1149 = 0U;
    for (; i_kpr1149 < hi_kpr1150; i_kpr1149++) {
        uint32_t j_kpr1151 = i_kpr1149;
        elems_tile0[j_kpr1151 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1151 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1151 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1151 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1154 = 0U;
    for (; i_kpr1154 < 2U; i_kpr1154++) {
        uint32_t j_kpr1155 = i_kpr1154;
        uint32_t off = n_idx + j_kpr1155 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1152 = 0U;
            for (; i_kpr1152 < 1U; i_kpr1152++) {
                uint32_t j_kpr1153 = i_kpr1152;
                uint32_t start = j_kpr1155 * 4U + j_kpr1153 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1153 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1156 = 0U;
        for (; i_kpr1156 < 4U; i_kpr1156++) {
            uint32_t j_kpr1157 = i_kpr1156;
            vec_memcpy(elems_tile0 + (j_kpr1157 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1157 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1158 = 0U;
        for (; i_kpr1158 < 4U; i_kpr1158++) {
            uint32_t j_kpr1159 = i_kpr1158;
            vec_memcpy(col_ind_tile0 + (j_kpr1159 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1159 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1160 = 0U;
        for (; i_kpr1160 < to_; i_kpr1160++)
            elems_tile0[i_kpr1160 * 32U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1162 = 0U;
            for (; i_kpr1162 < 4U; i_kpr1162++) {
                uint32_t j_kpr1163 = i_kpr1162;
                vec_memcpy(elems_tile0 + (j_kpr1163 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1163 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1164 = 0U;
            for (; i_kpr1164 < 4U; i_kpr1164++) {
                uint32_t j_kpr1165 = i_kpr1164;
                vec_memcpy(col_ind_tile0 + (j_kpr1165 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1165 * 32U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1167 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1166 = 0U;
    for (; i_kpr1166 < hi_kpr1167; i_kpr1166++) {
        uint32_t j_kpr1168 = i_kpr1166;
        elems_tile0[j_kpr1168 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1168 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1168 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1168 * 32U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1171 = 0U;
    for (; i_kpr1171 < 1U; i_kpr1171++) {
        uint32_t j_kpr1172 = i_kpr1171;
        uint32_t off = n_idx + j_kpr1172 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1169 = 0U;
            for (; i_kpr1169 < 1U; i_kpr1169++) {
                uint32_t j_kpr1170 = i_kpr1169;
                uint32_t start = j_kpr1172 * 4U + j_kpr1170 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1170 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1173 = 0U;
        for (; i_kpr1173 < 4U; i_kpr1173++) {
            uint32_t j_kpr1174 = i_kpr1173;
            vec_memcpy(elems_tile0 + (j_kpr1174 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1174 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1175 = 0U;
        for (; i_kpr1175 < 4U; i_kpr1175++) {
            uint32_t j_kpr1176 = i_kpr1175;
            vec_memcpy(col_ind_tile0 + (j_kpr1176 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1176 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1177 = 0U;
        for (; i_kpr1177 < to_; i_kpr1177++)
            elems_tile0[i_kpr1177 * 32U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1179 = 0U;
            for (; i_kpr1179 < 4U; i_kpr1179++) {
                uint32_t j_kpr1180 = i_kpr1179;
                vec_memcpy(elems_tile0 + (j_kpr1180 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1180 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1181 = 0U;
            for (; i_kpr1181 < 4U; i_kpr1181++) {
                uint32_t j_kpr1182 = i_kpr1181;
                vec_memcpy(col_ind_tile0 + (j_kpr1182 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1182 * 32U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1184 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1183 = 0U;
    for (; i_kpr1183 < hi_kpr1184; i_kpr1183++) {
        uint32_t j_kpr1185 = i_kpr1183;
        elems_tile0[j_kpr1185 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1185 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1185 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1185 * 32U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1188 = 0U;
    for (; i_kpr1188 < 1U; i_kpr1188++) {
        uint32_t j_kpr1189 = i_kpr1188;
        uint32_t off = n_idx + j_kpr1189 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1186 = 0U;
            for (; i_kpr1186 < 1U; i_kpr1186++) {
                uint32_t j_kpr1187 = i_kpr1186;
                uint32_t start = j_kpr1189 * 4U + j_kpr1187 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1187 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1190 = 0U;
        for (; i_kpr1190 < 8U; i_kpr1190++) {
            uint32_t j_kpr1191 = i_kpr1190;
            vec_memcpy(elems_tile0 + (j_kpr1191 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1191 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1192 = 0U;
        for (; i_kpr1192 < 8U; i_kpr1192++) {
            uint32_t j_kpr1193 = i_kpr1192;
            vec_memcpy(col_ind_tile0 + (j_kpr1193 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1193 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1194 = 0U;
        for (; i_kpr1194 < to_; i_kpr1194++)
            elems_tile0[i_kpr1194 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1196 = 0U;
            for (; i_kpr1196 < 8U; i_kpr1196++) {
                uint32_t j_kpr1197 = i_kpr1196;
                vec_memcpy(elems_tile0 + (j_kpr1197 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1197 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1198 = 0U;
            for (; i_kpr1198 < 8U; i_kpr1198++) {
                uint32_t j_kpr1199 = i_kpr1198;
                vec_memcpy(col_ind_tile0 + (j_kpr1199 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1199 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1201 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1200 = 0U;
    for (; i_kpr1200 < hi_kpr1201; i_kpr1200++) {
        uint32_t j_kpr1202 = i_kpr1200;
        elems_tile0[j_kpr1202 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1202 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1202 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1202 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1205 = 0U;
    for (; i_kpr1205 < 4U; i_kpr1205++) {
        uint32_t j_kpr1206 = i_kpr1205;
        uint32_t off = n_idx + j_kpr1206 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1203 = 0U;
            for (; i_kpr1203 < 1U; i_kpr1203++) {
                uint32_t j_kpr1204 = i_kpr1203;
                uint32_t start = j_kpr1206 * 4U + j_kpr1204 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1204 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1207 = 0U;
        for (; i_kpr1207 < 8U; i_kpr1207++) {
            uint32_t j_kpr1208 = i_kpr1207;
            vec_memcpy(elems_tile0 + (j_kpr1208 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1208 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1209 = 0U;
        for (; i_kpr1209 < 8U; i_kpr1209++) {
            uint32_t j_kpr1210 = i_kpr1209;
            vec_memcpy(col_ind_tile0 + (j_kpr1210 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1210 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1211 = 0U;
        for (; i_kpr1211 < to_; i_kpr1211++)
            elems_tile0[i_kpr1211 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1213 = 0U;
            for (; i_kpr1213 < 8U; i_kpr1213++) {
                uint32_t j_kpr1214 = i_kpr1213;
                vec_memcpy(elems_tile0 + (j_kpr1214 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1214 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1215 = 0U;
            for (; i_kpr1215 < 8U; i_kpr1215++) {
                uint32_t j_kpr1216 = i_kpr1215;
                vec_memcpy(col_ind_tile0 + (j_kpr1216 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1216 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1218 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1217 = 0U;
    for (; i_kpr1217 < hi_kpr1218; i_kpr1217++) {
        uint32_t j_kpr1219 = i_kpr1217;
        elems_tile0[j_kpr1219 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1219 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1219 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1219 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1222 = 0U;
    for (; i_kpr1222 < 4U; i_kpr1222++) {
        uint32_t j_kpr1223 = i_kpr1222;
        uint32_t off = n_idx + j_kpr1223 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1220 = 0U;
            for (; i_kpr1220 < 1U; i_kpr1220++) {
                uint32_t j_kpr1221 = i_kpr1220;
                uint32_t start = j_kpr1223 * 4U + j_kpr1221 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1221 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1224 = 0U;
        for (; i_kpr1224 < 4U; i_kpr1224++) {
            uint32_t j_kpr1225 = i_kpr1224;
            vec_memcpy(elems_tile0 + (j_kpr1225 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1225 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1226 = 0U;
        for (; i_kpr1226 < 4U; i_kpr1226++) {
            uint32_t j_kpr1227 = i_kpr1226;
            vec_memcpy(col_ind_tile0 + (j_kpr1227 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1227 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1228 = 0U;
        for (; i_kpr1228 < to_; i_kpr1228++)
            elems_tile0[i_kpr1228 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1230 = 0U;
            for (; i_kpr1230 < 4U; i_kpr1230++) {
                uint32_t j_kpr1231 = i_kpr1230;
                vec_memcpy(elems_tile0 + (j_kpr1231 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1231 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1232 = 0U;
            for (; i_kpr1232 < 4U; i_kpr1232++) {
                uint32_t j_kpr1233 = i_kpr1232;
                vec_memcpy(col_ind_tile0 + (j_kpr1233 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1233 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1235 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1234 = 0U;
    for (; i_kpr1234 < hi_kpr1235; i_kpr1234++) {
        uint32_t j_kpr1236 = i_kpr1234;
        elems_tile0[j_kpr1236 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1236 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1236 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1236 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1239 = 0U;
    for (; i_kpr1239 < 2U; i_kpr1239++) {
        uint32_t j_kpr1240 = i_kpr1239;
        uint32_t off = n_idx + j_kpr1240 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1237 = 0U;
            for (; i_kpr1237 < 1U; i_kpr1237++) {
                uint32_t j_kpr1238 = i_kpr1237;
                uint32_t start = j_kpr1240 * 4U + j_kpr1238 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1238 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1241 = 0U;
        for (; i_kpr1241 < 4U; i_kpr1241++) {
            uint32_t j_kpr1242 = i_kpr1241;
            vec_memcpy(elems_tile0 + (j_kpr1242 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1242 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1243 = 0U;
        for (; i_kpr1243 < 4U; i_kpr1243++) {
            uint32_t j_kpr1244 = i_kpr1243;
            vec_memcpy(col_ind_tile0 + (j_kpr1244 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1244 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1245 = 0U;
        for (; i_kpr1245 < to_; i_kpr1245++)
            elems_tile0[i_kpr1245 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1247 = 0U;
            for (; i_kpr1247 < 4U; i_kpr1247++) {
                uint32_t j_kpr1248 = i_kpr1247;
                vec_memcpy(elems_tile0 + (j_kpr1248 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1248 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1249 = 0U;
            for (; i_kpr1249 < 4U; i_kpr1249++) {
                uint32_t j_kpr1250 = i_kpr1249;
                vec_memcpy(col_ind_tile0 + (j_kpr1250 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1250 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1252 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1251 = 0U;
    for (; i_kpr1251 < hi_kpr1252; i_kpr1251++) {
        uint32_t j_kpr1253 = i_kpr1251;
        elems_tile0[j_kpr1253 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1253 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1253 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1253 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1256 = 0U;
    for (; i_kpr1256 < 2U; i_kpr1256++) {
        uint32_t j_kpr1257 = i_kpr1256;
        uint32_t off = n_idx + j_kpr1257 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1254 = 0U;
            for (; i_kpr1254 < 1U; i_kpr1254++) {
                uint32_t j_kpr1255 = i_kpr1254;
                uint32_t start = j_kpr1257 * 4U + j_kpr1255 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1255 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1258 = 0U;
        for (; i_kpr1258 < 2U; i_kpr1258++) {
            uint32_t j_kpr1259 = i_kpr1258;
            vec_memcpy(elems_tile0 + (j_kpr1259 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1259 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1260 = 0U;
        for (; i_kpr1260 < 2U; i_kpr1260++) {
            uint32_t j_kpr1261 = i_kpr1260;
            vec_memcpy(col_ind_tile0 + (j_kpr1261 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1261 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1262 = 0U;
        for (; i_kpr1262 < to_; i_kpr1262++)
            elems_tile0[i_kpr1262 * 64U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1264 = 0U;
            for (; i_kpr1264 < 2U; i_kpr1264++) {
                uint32_t j_kpr1265 = i_kpr1264;
                vec_memcpy(elems_tile0 + (j_kpr1265 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1265 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1266 = 0U;
            for (; i_kpr1266 < 2U; i_kpr1266++) {
                uint32_t j_kpr1267 = i_kpr1266;
                vec_memcpy(col_ind_tile0 + (j_kpr1267 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1267 * 64U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1269 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1268 = 0U;
    for (; i_kpr1268 < hi_kpr1269; i_kpr1268++) {
        uint32_t j_kpr1270 = i_kpr1268;
        elems_tile0[j_kpr1270 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1270 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1270 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1270 * 64U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1273 = 0U;
    for (; i_kpr1273 < 1U; i_kpr1273++) {
        uint32_t j_kpr1274 = i_kpr1273;
        uint32_t off = n_idx + j_kpr1274 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr1271 = 0U;
            for (; i_kpr1271 < 1U; i_kpr1271++) {
                uint32_t j_kpr1272 = i_kpr1271;
                uint32_t start = j_kpr1274 * 4U + j_kpr1272 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1272 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1275 = 0U;
        for (; i_kpr1275 < 2U; i_kpr1275++) {
            uint32_t j_kpr1276 = i_kpr1275;
            vec_memcpy(elems_tile0 + (j_kpr1276 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1276 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1277 = 0U;
        for (; i_kpr1277 < 2U; i_kpr1277++) {
            uint32_t j_kpr1278 = i_kpr1277;
            vec_memcpy(col_ind_tile0 + (j_kpr1278 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1278 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1279 = 0U;
        for (; i_kpr1279 < to_; i_kpr1279++)
            elems_tile0[i_kpr1279 * 64U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1281 = 0U;
            for (; i_kpr1281 < 2U; i_kpr1281++) {
                uint32_t j_kpr1282 = i_kpr1281;
                vec_memcpy(elems_tile0 + (j_kpr1282 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1282 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1283 = 0U;
            for (; i_kpr1283 < 2U; i_kpr1283++) {
                uint32_t j_kpr1284 = i_kpr1283;
                vec_memcpy(col_ind_tile0 + (j_kpr1284 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1284 * 64U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf03 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1286 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1285 = 0U;
    for (; i_kpr1285 < hi_kpr1286; i_kpr1285++) {
        uint32_t j_kpr1287 = i_kpr1285;
        elems_tile0[j_kpr1287 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1287 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1287 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1287 * 64U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1290 = 0U;
    for (; i_kpr1290 < 1U; i_kpr1290++) {
        uint32_t j_kpr1291 = i_kpr1290;
        uint32_t off = n_idx + j_kpr1291 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr1288 = 0U;
            for (; i_kpr1288 < 1U; i_kpr1288++) {
                uint32_t j_kpr1289 = i_kpr1288;
                uint32_t start = j_kpr1291 * 4U + j_kpr1289 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1289 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1292 = 0U;
        for (; i_kpr1292 < 8U; i_kpr1292++) {
            uint32_t j_kpr1293 = i_kpr1292;
            vec_memcpy(elems_tile0 + (j_kpr1293 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1293 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1294 = 0U;
        for (; i_kpr1294 < 8U; i_kpr1294++) {
            uint32_t j_kpr1295 = i_kpr1294;
            vec_memcpy(col_ind_tile0 + (j_kpr1295 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1295 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1296 = 0U;
        for (; i_kpr1296 < to_; i_kpr1296++)
            elems_tile0[i_kpr1296 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1298 = 0U;
            for (; i_kpr1298 < 8U; i_kpr1298++) {
                uint32_t j_kpr1299 = i_kpr1298;
                vec_memcpy(elems_tile0 + (j_kpr1299 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1299 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1300 = 0U;
            for (; i_kpr1300 < 8U; i_kpr1300++) {
                uint32_t j_kpr1301 = i_kpr1300;
                vec_memcpy(col_ind_tile0 + (j_kpr1301 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1301 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1303 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1302 = 0U;
    for (; i_kpr1302 < hi_kpr1303; i_kpr1302++) {
        uint32_t j_kpr1304 = i_kpr1302;
        elems_tile0[j_kpr1304 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1304 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1304 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1304 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1307 = 0U;
    for (; i_kpr1307 < 8U; i_kpr1307++) {
        uint32_t j_kpr1308 = i_kpr1307;
        uint32_t off = n_idx + j_kpr1308 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1305 = 0U;
            for (; i_kpr1305 < 1U; i_kpr1305++) {
                uint32_t j_kpr1306 = i_kpr1305;
                uint32_t start = j_kpr1308 * 4U + j_kpr1306 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1306 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1309 = 0U;
        for (; i_kpr1309 < 8U; i_kpr1309++) {
            uint32_t j_kpr1310 = i_kpr1309;
            vec_memcpy(elems_tile0 + (j_kpr1310 * 16U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1310 * 16U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1311 = 0U;
        for (; i_kpr1311 < 8U; i_kpr1311++) {
            uint32_t j_kpr1312 = i_kpr1311;
            vec_memcpy(col_ind_tile0 + (j_kpr1312 * 16U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1312 * 16U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 15U - threadIdx.x) / 16U;
        uint32_t i_kpr1313 = 0U;
        for (; i_kpr1313 < to_; i_kpr1313++)
            elems_tile0[i_kpr1313 * 16U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 16U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1315 = 0U;
            for (; i_kpr1315 < 8U; i_kpr1315++) {
                uint32_t j_kpr1316 = i_kpr1315;
                vec_memcpy(elems_tile0 + (j_kpr1316 * 16U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1316 * 16U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1317 = 0U;
            for (; i_kpr1317 < 8U; i_kpr1317++) {
                uint32_t j_kpr1318 = i_kpr1317;
                vec_memcpy(col_ind_tile0 + (j_kpr1318 * 16U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1318 * 16U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 16U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1320 = (__anf01 + 15U - threadIdx.x) / 16U;
    uint32_t i_kpr1319 = 0U;
    for (; i_kpr1319 < hi_kpr1320; i_kpr1319++) {
        uint32_t j_kpr1321 = i_kpr1319;
        elems_tile0[j_kpr1321 * 16U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1321 * 16U + threadIdx.x];
        col_ind_tile0[j_kpr1321 * 16U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1321 * 16U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 16U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1324 = 0U;
    for (; i_kpr1324 < 8U; i_kpr1324++) {
        uint32_t j_kpr1325 = i_kpr1324;
        uint32_t off = n_idx + j_kpr1325 * 16U * 4U;
        if (off < cols) {
            uint32_t i_kpr1322 = 0U;
            for (; i_kpr1322 < 1U; i_kpr1322++) {
                uint32_t j_kpr1323 = i_kpr1322;
                uint32_t start = j_kpr1325 * 4U + j_kpr1323 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1323 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1326 = 0U;
        for (; i_kpr1326 < 4U; i_kpr1326++) {
            uint32_t j_kpr1327 = i_kpr1326;
            vec_memcpy(elems_tile0 + (j_kpr1327 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1327 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1328 = 0U;
        for (; i_kpr1328 < 4U; i_kpr1328++) {
            uint32_t j_kpr1329 = i_kpr1328;
            vec_memcpy(col_ind_tile0 + (j_kpr1329 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1329 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1330 = 0U;
        for (; i_kpr1330 < to_; i_kpr1330++)
            elems_tile0[i_kpr1330 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1332 = 0U;
            for (; i_kpr1332 < 4U; i_kpr1332++) {
                uint32_t j_kpr1333 = i_kpr1332;
                vec_memcpy(elems_tile0 + (j_kpr1333 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1333 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1334 = 0U;
            for (; i_kpr1334 < 4U; i_kpr1334++) {
                uint32_t j_kpr1335 = i_kpr1334;
                vec_memcpy(col_ind_tile0 + (j_kpr1335 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1335 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1337 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1336 = 0U;
    for (; i_kpr1336 < hi_kpr1337; i_kpr1336++) {
        uint32_t j_kpr1338 = i_kpr1336;
        elems_tile0[j_kpr1338 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1338 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1338 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1338 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1341 = 0U;
    for (; i_kpr1341 < 4U; i_kpr1341++) {
        uint32_t j_kpr1342 = i_kpr1341;
        uint32_t off = n_idx + j_kpr1342 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1339 = 0U;
            for (; i_kpr1339 < 1U; i_kpr1339++) {
                uint32_t j_kpr1340 = i_kpr1339;
                uint32_t start = j_kpr1342 * 4U + j_kpr1340 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1340 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1343 = 0U;
        for (; i_kpr1343 < 4U; i_kpr1343++) {
            uint32_t j_kpr1344 = i_kpr1343;
            vec_memcpy(elems_tile0 + (j_kpr1344 * 32U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1344 * 32U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1345 = 0U;
        for (; i_kpr1345 < 4U; i_kpr1345++) {
            uint32_t j_kpr1346 = i_kpr1345;
            vec_memcpy(col_ind_tile0 + (j_kpr1346 * 32U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1346 * 32U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 31U - threadIdx.x) / 32U;
        uint32_t i_kpr1347 = 0U;
        for (; i_kpr1347 < to_; i_kpr1347++)
            elems_tile0[i_kpr1347 * 32U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 32U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1349 = 0U;
            for (; i_kpr1349 < 4U; i_kpr1349++) {
                uint32_t j_kpr1350 = i_kpr1349;
                vec_memcpy(elems_tile0 + (j_kpr1350 * 32U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1350 * 32U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1351 = 0U;
            for (; i_kpr1351 < 4U; i_kpr1351++) {
                uint32_t j_kpr1352 = i_kpr1351;
                vec_memcpy(col_ind_tile0 + (j_kpr1352 * 32U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1352 * 32U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 32U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1354 = (__anf01 + 31U - threadIdx.x) / 32U;
    uint32_t i_kpr1353 = 0U;
    for (; i_kpr1353 < hi_kpr1354; i_kpr1353++) {
        uint32_t j_kpr1355 = i_kpr1353;
        elems_tile0[j_kpr1355 * 32U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1355 * 32U + threadIdx.x];
        col_ind_tile0[j_kpr1355 * 32U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1355 * 32U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 32U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1358 = 0U;
    for (; i_kpr1358 < 4U; i_kpr1358++) {
        uint32_t j_kpr1359 = i_kpr1358;
        uint32_t off = n_idx + j_kpr1359 * 32U * 4U;
        if (off < cols) {
            uint32_t i_kpr1356 = 0U;
            for (; i_kpr1356 < 1U; i_kpr1356++) {
                uint32_t j_kpr1357 = i_kpr1356;
                uint32_t start = j_kpr1359 * 4U + j_kpr1357 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1357 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1360 = 0U;
        for (; i_kpr1360 < 2U; i_kpr1360++) {
            uint32_t j_kpr1361 = i_kpr1360;
            vec_memcpy(elems_tile0 + (j_kpr1361 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1361 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1362 = 0U;
        for (; i_kpr1362 < 2U; i_kpr1362++) {
            uint32_t j_kpr1363 = i_kpr1362;
            vec_memcpy(col_ind_tile0 + (j_kpr1363 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1363 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1364 = 0U;
        for (; i_kpr1364 < to_; i_kpr1364++)
            elems_tile0[i_kpr1364 * 64U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1366 = 0U;
            for (; i_kpr1366 < 2U; i_kpr1366++) {
                uint32_t j_kpr1367 = i_kpr1366;
                vec_memcpy(elems_tile0 + (j_kpr1367 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1367 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1368 = 0U;
            for (; i_kpr1368 < 2U; i_kpr1368++) {
                uint32_t j_kpr1369 = i_kpr1368;
                vec_memcpy(col_ind_tile0 + (j_kpr1369 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1369 * 64U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 64U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1371 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1370 = 0U;
    for (; i_kpr1370 < hi_kpr1371; i_kpr1370++) {
        uint32_t j_kpr1372 = i_kpr1370;
        elems_tile0[j_kpr1372 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1372 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1372 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1372 * 64U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1375 = 0U;
    for (; i_kpr1375 < 2U; i_kpr1375++) {
        uint32_t j_kpr1376 = i_kpr1375;
        uint32_t off = n_idx + j_kpr1376 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr1373 = 0U;
            for (; i_kpr1373 < 1U; i_kpr1373++) {
                uint32_t j_kpr1374 = i_kpr1373;
                uint32_t start = j_kpr1376 * 4U + j_kpr1374 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1374 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1377 = 0U;
        for (; i_kpr1377 < 2U; i_kpr1377++) {
            uint32_t j_kpr1378 = i_kpr1377;
            vec_memcpy(elems_tile0 + (j_kpr1378 * 64U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1378 * 64U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1379 = 0U;
        for (; i_kpr1379 < 2U; i_kpr1379++) {
            uint32_t j_kpr1380 = i_kpr1379;
            vec_memcpy(col_ind_tile0 + (j_kpr1380 * 64U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1380 * 64U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 63U - threadIdx.x) / 64U;
        uint32_t i_kpr1381 = 0U;
        for (; i_kpr1381 < to_; i_kpr1381++)
            elems_tile0[i_kpr1381 * 64U + threadIdx.x] = 0.0f;
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB + (cols * kr + (n_idx + __anf01 * 64U * 4U) +
                                         kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
            uint32_t i_kpr1383 = 0U;
            for (; i_kpr1383 < 2U; i_kpr1383++) {
                uint32_t j_kpr1384 = i_kpr1383;
                vec_memcpy(elems_tile0 + (j_kpr1384 * 64U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1384 * 64U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1385 = 0U;
            for (; i_kpr1385 < 2U; i_kpr1385++) {
                uint32_t j_kpr1386 = i_kpr1385;
                vec_memcpy(col_ind_tile0 + (j_kpr1386 * 64U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1386 * 64U + threadIdx.x) * 4U));
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
                            uint32_t k2 = 0U;
                            while (k2 < 1U) {
                                uint32_t kv1 = k2;
                                vec_memcpy(lchunk + kv1 * 4U,
                                    gB + (cols * kr +
                                             (n_idx + __anf03 * 64U * 4U) +
                                             kv1 * 4U));
                                k2 = kv1 + 1U;
                            }
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
    uint32_t hi_kpr1388 = (__anf01 + 63U - threadIdx.x) / 64U;
    uint32_t i_kpr1387 = 0U;
    for (; i_kpr1387 < hi_kpr1388; i_kpr1387++) {
        uint32_t j_kpr1389 = i_kpr1387;
        elems_tile0[j_kpr1389 * 64U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1389 * 64U + threadIdx.x];
        col_ind_tile0[j_kpr1389 * 64U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1389 * 64U + threadIdx.x];
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf02 * 64U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
                    uint32_t ix = 0U;
                    for (; ix < 4U; ix++)
                        out[__anf1 * 4U + ix] += kx * lchunk[ix];
                }
            }
        }
    }
    uint32_t i_kpr1392 = 0U;
    for (; i_kpr1392 < 2U; i_kpr1392++) {
        uint32_t j_kpr1393 = i_kpr1392;
        uint32_t off = n_idx + j_kpr1393 * 64U * 4U;
        if (off < cols) {
            uint32_t i_kpr1390 = 0U;
            for (; i_kpr1390 < 1U; i_kpr1390++) {
                uint32_t j_kpr1391 = i_kpr1390;
                uint32_t start = j_kpr1393 * 4U + j_kpr1391 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1391 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1394 = 0U;
        for (; i_kpr1394 < 1U; i_kpr1394++) {
            uint32_t j_kpr1395 = i_kpr1394;
            vec_memcpy(elems_tile0 + (j_kpr1395 * 128U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1395 * 128U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1396 = 0U;
        for (; i_kpr1396 < 1U; i_kpr1396++) {
            uint32_t j_kpr1397 = i_kpr1396;
            vec_memcpy(col_ind_tile0 + (j_kpr1397 * 128U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1397 * 128U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i_kpr1398 = 0U;
        for (; i_kpr1398 < to_; i_kpr1398++)
            elems_tile0[i_kpr1398 * 128U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 128U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1400 = 0U;
            for (; i_kpr1400 < 1U; i_kpr1400++) {
                uint32_t j_kpr1401 = i_kpr1400;
                vec_memcpy(elems_tile0 + (j_kpr1401 * 128U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1401 * 128U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1402 = 0U;
            for (; i_kpr1402 < 1U; i_kpr1402++) {
                uint32_t j_kpr1403 = i_kpr1402;
                vec_memcpy(
                    col_ind_tile0 + (j_kpr1403 * 128U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1403 * 128U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 128U * 4U) +
                                        kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1405 = (__anf01 + 127U - threadIdx.x) / 128U;
    uint32_t i_kpr1404 = 0U;
    for (; i_kpr1404 < hi_kpr1405; i_kpr1404++) {
        uint32_t j_kpr1406 = i_kpr1404;
        elems_tile0[j_kpr1406 * 128U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1406 * 128U + threadIdx.x];
        col_ind_tile0[j_kpr1406 * 128U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1406 * 128U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 128U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1409 = 0U;
    for (; i_kpr1409 < 1U; i_kpr1409++) {
        uint32_t j_kpr1410 = i_kpr1409;
        uint32_t off = n_idx + j_kpr1410 * 128U * 4U;
        if (off < cols) {
            uint32_t i_kpr1407 = 0U;
            for (; i_kpr1407 < 1U; i_kpr1407++) {
                uint32_t j_kpr1408 = i_kpr1407;
                uint32_t start = j_kpr1410 * 4U + j_kpr1408 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1408 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
        uint32_t i_kpr1411 = 0U;
        for (; i_kpr1411 < 1U; i_kpr1411++) {
            uint32_t j_kpr1412 = i_kpr1411;
            vec_memcpy(elems_tile0 + (j_kpr1412 * 128U + threadIdx.x) * 4U,
                gA.elems + (ri_ + (j_kpr1412 * 128U + threadIdx.x) * 4U));
        }
        uint32_t i_kpr1413 = 0U;
        for (; i_kpr1413 < 1U; i_kpr1413++) {
            uint32_t j_kpr1414 = i_kpr1413;
            vec_memcpy(col_ind_tile0 + (j_kpr1414 * 128U + threadIdx.x) * 4U,
                gA.col_ind + (ri_ + (j_kpr1414 * 128U + threadIdx.x) * 4U));
        }
        __syncthreads();
        uint32_t to_ = (ri - ri_ + 127U - threadIdx.x) / 128U;
        uint32_t i_kpr1415 = 0U;
        for (; i_kpr1415 < to_; i_kpr1415++)
            elems_tile0[i_kpr1415 * 128U + threadIdx.x] = 0.0f;
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
                    uint32_t k2 = 0U;
                    while (k2 < 1U) {
                        uint32_t kv1 = k2;
                        vec_memcpy(lchunk + kv1 * 4U,
                            gB + (cols * kr + (n_idx + __anf01 * 128U * 4U) +
                                     kv1 * 4U));
                        k2 = kv1 + 1U;
                    }
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
            uint32_t i_kpr1417 = 0U;
            for (; i_kpr1417 < 1U; i_kpr1417++) {
                uint32_t j_kpr1418 = i_kpr1417;
                vec_memcpy(elems_tile0 + (j_kpr1418 * 128U + threadIdx.x) * 4U,
                    gA.elems + (off + (j_kpr1418 * 128U + threadIdx.x) * 4U));
            }
            uint32_t i_kpr1419 = 0U;
            for (; i_kpr1419 < 1U; i_kpr1419++) {
                uint32_t j_kpr1420 = i_kpr1419;
                vec_memcpy(
                    col_ind_tile0 + (j_kpr1420 * 128U + threadIdx.x) * 4U,
                    gA.col_ind + (off + (j_kpr1420 * 128U + threadIdx.x) * 4U));
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
                        uint32_t k2 = 0U;
                        while (k2 < 1U) {
                            uint32_t kv1 = k2;
                            vec_memcpy(lchunk + kv1 * 4U,
                                gB +
                                    (cols * kr + (n_idx + __anf03 * 128U * 4U) +
                                        kv1 * 4U));
                            k2 = kv1 + 1U;
                        }
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
    uint32_t hi_kpr1422 = (__anf01 + 127U - threadIdx.x) / 128U;
    uint32_t i_kpr1421 = 0U;
    for (; i_kpr1421 < hi_kpr1422; i_kpr1421++) {
        uint32_t j_kpr1423 = i_kpr1421;
        elems_tile0[j_kpr1423 * 128U + threadIdx.x] =
            gA.elems[re - __anf01 + j_kpr1423 * 128U + threadIdx.x];
        col_ind_tile0[j_kpr1423 * 128U + threadIdx.x] =
            gA.col_ind[re - __anf01 + j_kpr1423 * 128U + threadIdx.x];
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
                uint32_t k2 = 0U;
                while (k2 < 1U) {
                    uint32_t kv1 = k2;
                    vec_memcpy(lchunk + kv1 * 4U,
                        gB + (cols * kr + (n_idx + __anf02 * 128U * 4U) +
                                 kv1 * 4U));
                    k2 = kv1 + 1U;
                }
                uint32_t ix = 0U;
                for (; ix < 4U; ix++)
                    out[__anf1 * 4U + ix] += kx * lchunk[ix];
            }
        }
    }
    uint32_t i_kpr1426 = 0U;
    for (; i_kpr1426 < 1U; i_kpr1426++) {
        uint32_t j_kpr1427 = i_kpr1426;
        uint32_t off = n_idx + j_kpr1427 * 128U * 4U;
        if (off < cols) {
            uint32_t i_kpr1424 = 0U;
            for (; i_kpr1424 < 1U; i_kpr1424++) {
                uint32_t j_kpr1425 = i_kpr1424;
                uint32_t start = j_kpr1427 * 4U + j_kpr1425 * 4U;
                uint32_t address = cols * m_idx + (off + j_kpr1425 * 4U);
                float combined[4U];
                memset(combined, 0U, 4U * sizeof(float));
                vec_memcpy(combined, gC + address);
                uint32_t lane = 0U;
                for (; lane < 4U; lane++)
                    combined[lane] = out[start + lane];
                vec_memcpy(gC + address, combined);
            }
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
