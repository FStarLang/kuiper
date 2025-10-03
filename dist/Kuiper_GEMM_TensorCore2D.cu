
#include "Kuiper_GEMM_TensorCore2D.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x4
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 16U;
            uint32_t col = i2 % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 1024U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 64U;
            uint32_t col = i % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  16U * (threadIdx.x / 32U /
                                                         1U * 64U) +
                                                  __anf04 * 16U +
                                                  16U * (i0 * 16U)), 16U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  64U * (__anf05 * 16U) +
                                                  0U * 64U + i1 * 16U), 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 4U; j += 1U) {
            auto & __anf2 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 64U) * 64U) +
                                               blockIdx.x % (cols / 64U) * 64U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       64U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_KCALL(__hoisted_0, rows / 64U * (cols / 64U), 32U, 4096U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_32x8x16_1x2
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           32,
                                                           8,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           32,
                                                           8,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16), 2U));
    uint32_t fi = 0U;
    for (; fi < 2U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 1024U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 32U;
            uint32_t col = i % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         2U * 32U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 32U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  32U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 2U * 16U +
                                                  i1 * 8U), 32U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 1U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 1U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 32U) * 32U) +
                                               blockIdx.x % (cols / 32U) * 32U +
                                               cols * (threadIdx.x / 32U / 2U *
                                                       32U)
                                               + threadIdx.x / 32U % 2U * 16U +
                                               cols * (i * 32U)
                                               + j * 8U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_32x8x16_1x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half_t *gA,
                                                             half_t *gB,
                                                             half_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_KCALL(__hoisted_1, rows / 32U * (cols / 32U), 64U, 4096U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_8x32x16_2x1
*/
static void __hoisted_2(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           8,
                                                           32,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           8,
                                                           32,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16), 2U));
    uint32_t fi = 0U;
    for (; fi < 2U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 1024U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 32U;
            uint32_t col = i % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         1U * 16U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 8U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 1U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  32U * (__anf05 * 16U) +
                                                  0U * 32U + i1 * 32U), 32U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 1U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 1U; j += 1U) {
            auto & __anf2 = accFrags[i + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 32U) * 32U) +
                                               blockIdx.x % (cols / 32U) * 32U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       16U)
                                               + cols * (i * 8U)
                                               + j * 32U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_8x32x16_2x1(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half_t *gA,
                                                             half_t *gB,
                                                             half_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_KCALL(__hoisted_2, rows / 32U * (cols / 32U), 64U, 4096U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x8x16_32x8x16
*/
static void __hoisted_3(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(1024U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 8U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           32,
                                                           8,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           32,
                                                           8,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16), 1U));
    uint32_t fi = 0U;
    for (; fi < 1U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 512U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 16U;
            uint32_t col = i2 % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 128U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 8U;
            uint32_t col = i % 8U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 8U + cols * row +
                       col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 8U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  16U * (threadIdx.x / 32U /
                                                         1U * 32U) +
                                                  __anf04 * 16U +
                                                  16U * (i0 * 32U)), 16U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 1U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  8U * (__anf05 * 16U) +
                                                  0U * 8U + i1 * 8U), 8U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 1U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 1U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 1U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 1U; j += 1U) {
            auto & __anf2 = accFrags[i + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 8U) * 32U) +
                                               blockIdx.x % (cols / 8U) * 8U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       32U)
                                               + cols * (i * 32U)
                                               + j * 8U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x8x16_32x8x16(uint32_t rows,
                                                        uint32_t shared,
                                                        uint32_t cols,
                                                        half_t *gA,
                                                        half_t *gB, half_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 8U == 0U);
    KPR_KCALL(__hoisted_3, rows / 32U * (cols / 8U), 32U, 1280U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_8x32x16_8x32x16
*/
static void __hoisted_4(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           8,
                                                           32,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           8,
                                                           32,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16), 1U));
    uint32_t fi = 0U;
    for (; fi < 1U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 1024U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 32U;
            uint32_t col = i % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         1U * 8U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 8U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 1U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  32U * (__anf05 * 16U) +
                                                  0U * 32U + i1 * 32U), 32U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 1U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 1U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 1U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 1U; j += 1U) {
            auto & __anf2 = accFrags[i + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 32U) * 32U) +
                                               blockIdx.x % (cols / 32U) * 32U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       8U)
                                               + cols * (i * 8U)
                                               + j * 32U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_8x32x16_8x32x16(uint32_t rows,
                                                        uint32_t shared,
                                                        uint32_t cols,
                                                        half_t *gA,
                                                        half_t *gB, half_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_KCALL(__hoisted_4, rows / 32U * (cols / 32U), 128U, 4096U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_16x16x16
*/
static void __hoisted_5(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(512U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 16U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         1U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  1U));
    uint32_t fi = 0U;
    for (; fi < 1U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 256U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 16U;
            uint32_t col = i2 % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 16U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 256U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 16U;
            uint32_t col = i % 16U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 16U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 16U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 1U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  16U * (threadIdx.x / 32U /
                                                         1U * 16U) +
                                                  __anf04 * 16U +
                                                  16U * (i0 * 16U)), 16U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 1U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  16U * (__anf05 * 16U) +
                                                  0U * 16U + i1 * 16U), 16U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 1U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 1U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 1U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 1U; j += 1U) {
            auto & __anf2 = accFrags[i + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 16U) * 16U) +
                                               blockIdx.x % (cols / 16U) * 16U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       16U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_16x16x16(uint32_t rows,
                                                          uint32_t shared,
                                                          uint32_t cols,
                                                          half_t *gA,
                                                          half_t *gB,
                                                          half_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    KPR_KCALL(__hoisted_5, rows / 16U * (cols / 16U), 32U, 1024U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x4
*/
static void __hoisted_6(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 64U;
            uint32_t col = i % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         1U * 64U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 16U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  64U * (__anf05 * 16U) +
                                                  0U * 64U + i1 * 16U), 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 4U; j += 1U) {
            auto & __anf2 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 64U) * 64U) +
                                               blockIdx.x % (cols / 64U) * 64U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       64U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_KCALL(__hoisted_6, rows / 64U * (cols / 64U), 32U, 16384U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_32x8x16_2x8
*/
static void __hoisted_7(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           32,
                                                           8,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           32,
                                                           8,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 64U;
            uint32_t col = i % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         1U * 64U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 32U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  64U * (__anf05 * 16U) +
                                                  0U * 64U + i1 * 8U), 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 8U; j += 1U) {
            auto & __anf2 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 64U) * 64U) +
                                               blockIdx.x % (cols / 64U) * 64U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       64U)
                                               + cols * (i * 32U)
                                               + j * 8U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_32x8x16_2x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half_t *gA,
                                                             half_t *gB,
                                                             half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_KCALL(__hoisted_7, rows / 64U * (cols / 64U), 32U, 16384U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_8x32x16_8x2
*/
static void __hoisted_8(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           8,
                                                           32,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           8,
                                                           32,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 64U;
            uint32_t col = i % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         1U * 64U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 8U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  64U * (__anf05 * 16U) +
                                                  0U * 64U + i1 * 32U), 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 64U) * 64U) +
                                               blockIdx.x % (cols / 64U) * 64U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       64U)
                                               + cols * (i * 8U)
                                               + j * 32U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_8x32x16_8x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half_t *gA,
                                                             half_t *gB,
                                                             half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_KCALL(__hoisted_8, rows / 64U * (cols / 64U), 32U, 16384U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x4
*/
static void __hoisted_9(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         2U * 64U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 2U * 64U +
                                                  i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 4U; j += 1U) {
            auto & __anf2 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 2U *
                                                       64U)
                                               + threadIdx.x / 32U % 2U * 64U +
                                               cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_9, rows / 128U * (cols / 128U), 128U, 16384U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x4
*/
static void __hoisted_10(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 8192U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 8192U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         2U * 64U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 16U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 2U * 64U +
                                                  i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 4U; j += 1U) {
            auto & __anf2 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 2U *
                                                       64U)
                                               + threadIdx.x / 32U % 2U * 64U +
                                               cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_10, rows / 128U * (cols / 128U), 128U, 32768U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x8
*/
static void __hoisted_11(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  64U));
    uint32_t fi = 0U;
    for (; fi < 64U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         1U * 128U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  0U * 128U + i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 8U; j += 1U) {
            auto & __anf2 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       128U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_11, rows / 128U * (cols / 128U), 32U, 16384U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x8
*/
static void __hoisted_12(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  64U));
    uint32_t fi = 0U;
    for (; fi < 64U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 8192U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 8192U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         1U * 128U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 16U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  0U * 128U + i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 8U; j += 1U) {
            auto & __anf2 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       128U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_12, rows / 128U * (cols / 128U), 32U, 32768U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_16x16x16_2x2
*/
static void __hoisted_13(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  4U));
    uint32_t fi = 0U;
    for (; fi < 4U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 1024U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 32U;
            uint32_t col = i % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         1U * 32U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  32U * (__anf05 * 16U) +
                                                  0U * 32U + i1 * 16U), 32U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 32U) * 32U) +
                                               blockIdx.x % (cols / 32U) * 32U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       32U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_KCALL(__hoisted_13, rows / 32U * (cols / 32U), 32U, 4096U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x2
*/
static void __hoisted_14(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  4U));
    uint32_t fi = 0U;
    for (; fi < 4U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 64U;
            uint32_t col = i % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         2U * 32U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 16U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  64U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 2U * 32U +
                                                  i1 * 16U), 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 64U) * 64U) +
                                               blockIdx.x % (cols / 64U) * 64U +
                                               cols * (threadIdx.x / 32U / 2U *
                                                       32U)
                                               + threadIdx.x / 32U % 2U * 32U +
                                               cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_KCALL(__hoisted_14, rows / 64U * (cols / 64U), 128U, 16384U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x2
*/
static void __hoisted_15(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  4U));
    uint32_t fi = 0U;
    for (; fi < 4U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         4U * 32U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 4U * 32U +
                                                  i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 4U *
                                                       32U)
                                               + threadIdx.x / 32U % 4U * 32U +
                                               cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_15, rows / 128U * (cols / 128U), 512U, 16384U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x2
*/
static void __hoisted_16(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  4U));
    uint32_t fi = 0U;
    for (; fi < 4U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 8192U; i2 += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 64U;
            uint32_t col = i2 % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 8192U; i += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  64U * (threadIdx.x / 32U /
                                                         4U * 32U) +
                                                  __anf04 * 16U +
                                                  64U * (i0 * 16U)), 64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 4U * 32U +
                                                  i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 4U *
                                                       32U)
                                               + threadIdx.x / 32U % 4U * 32U +
                                               cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_16, rows / 128U * (cols / 128U), 512U, 32768U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x8
*/
static void __hoisted_17(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  32U));
    uint32_t fi = 0U;
    for (; fi < 32U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         1U * 64U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  0U * 128U + i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 8U; j += 1U) {
            auto & __anf2 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       64U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_17, rows / 128U * (cols / 128U), 64U, 16384U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x4
*/
static void __hoisted_18(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  32U));
    uint32_t fi = 0U;
    for (; fi < 32U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         2U * 128U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  128U * (__anf05 * 16U) +
                                                  threadIdx.x / 32U % 2U * 64U +
                                                  i1 * 16U), 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 4U; j += 1U) {
            auto & __anf2 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 128U) * 128U) +
                                               blockIdx.x % (cols / 128U) *
                                               128U +
                                               cols * (threadIdx.x / 32U / 2U *
                                                       128U)
                                               + threadIdx.x / 32U % 2U * 64U +
                                               cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_18, rows / 128U * (cols / 128U), 64U, 16384U, shared,
              cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f32_32x32x32_16x16x16_2x2
*/
static void __hoisted_19(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         float_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(float, wmma::accumulator, 16, 16, 16),
                  4U));
    uint32_t fi = 0U;
    for (; fi < 4U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / 32U;
            uint32_t col = i2 % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < 1024U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / 32U;
            uint32_t col = i % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  32U * (threadIdx.x / 32U /
                                                         1U * 32U) +
                                                  __anf04 * 16U +
                                                  32U * (i0 * 16U)), 32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  32U * (__anf05 * 16U) +
                                                  0U * 32U + i1 * 16U), 32U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / 32U) * 32U) +
                                               blockIdx.x % (cols / 32U) * 32U +
                                               cols * (threadIdx.x / 32U / 1U *
                                                       32U)
                                               + cols * (i * 16U)
                                               + j * 16U),
                                    __anf2, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f32_32x32x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_KCALL(__hoisted_19, rows / 32U * (cols / 32U), 32U, 4096U, shared, cols,
              gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_2x2
*/
static void
__hoisted_20(uint32_t bm,
             uint32_t bn,
             uint32_t bk,
             uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U * (bm * bk));
    uint32_t num_k_tiles = shared / bk;
    uint32_t num_n_tiles = cols / bn;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t warpRow = threadIdx.x / 32U / (bn / 32U);
    uint32_t warpCol = threadIdx.x / 32U % (bn / 32U);
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         2U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  4U));
    uint32_t fi = 0U;
    for (; fi < 4U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < bm * bk; i2 += bm / 32U * (bn / 32U) * 32U * 8U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / bk;
            uint32_t col = i2 % bk;
            vec_memcpy(local,
                       tileA + shared * (mrow * bm) + __anf01 * bk +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * bk + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < bk * bn; i += bm / 32U * (bn / 32U) * 32U * 8U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / bn;
            uint32_t col = i % bn;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * bk) + mcol * bn + cols * row +
                       col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * bn + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < bk / 16U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  bk * (warpRow * 32U) +
                                                  __anf04 * 16U +
                                                  bk * (i0 * 16U)), bk);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  bn * (__anf05 * 16U) +
                                                  warpCol * 32U + i1 * 16U),
                                       bn);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 2U; j += 1U) {
            auto & __anf2 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / bn) * bm) +
                                               blockIdx.x % (cols / bn) * bn +
                                               cols * (threadIdx.x / 32U /
                                                       (bn / 32U) * 32U)
                                               +
                                               threadIdx.x / 32U % (bn / 32U) *
                                               32U + cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_2x2(uint32_t bm,
                                                     uint32_t bn,
                                                     uint32_t bk,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half_t *gA,
                                                     half_t *gB, half_t *gC)
{
    KPR_ASSERT(bm % 16U == 0U);
    KPR_ASSERT(bn % 16U == 0U);
    KPR_ASSERT(bk % 16U == 0U);
    KPR_GUARD(rows % bm == 0U);
    KPR_GUARD(shared % bk == 0U);
    KPR_GUARD(cols % bn == 0U);
    KPR_KCALL(__hoisted_20,
              rows / bm * (cols / bn),
              bm / 32U * (bn / 32U) * 32U,
              2U * (bm * bk) + 2U * (bk * bn),
              bm, bn, bk, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_4x4
*/
static void
__hoisted_21(uint32_t bm,
             uint32_t bn,
             uint32_t bk,
             uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U * (bm * bk));
    uint32_t num_k_tiles = shared / bk;
    uint32_t num_n_tiles = cols / bn;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t warpRow = threadIdx.x / 32U / (bn / 64U);
    uint32_t warpCol = threadIdx.x / 32U % (bn / 64U);
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         4U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  16U));
    uint32_t fi = 0U;
    for (; fi < 16U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < bm * bk; i2 += bm / 64U * (bn / 64U) * 32U * 8U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / bk;
            uint32_t col = i2 % bk;
            vec_memcpy(local,
                       tileA + shared * (mrow * bm) + __anf01 * bk +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * bk + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < bk * bn; i += bm / 64U * (bn / 64U) * 32U * 8U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / bn;
            uint32_t col = i % bn;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * bk) + mcol * bn + cols * row +
                       col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * bn + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < bk / 16U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  bk * (warpRow * 64U) +
                                                  __anf04 * 16U +
                                                  bk * (i0 * 16U)), bk);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  bn * (__anf05 * 16U) +
                                                  warpCol * 64U + i1 * 16U),
                                       bn);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 4U; j += 1U) {
            auto & __anf2 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / bn) * bm) +
                                               blockIdx.x % (cols / bn) * bn +
                                               cols * (threadIdx.x / 32U /
                                                       (bn / 64U) * 64U)
                                               +
                                               threadIdx.x / 32U % (bn / 64U) *
                                               64U + cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_4x4(uint32_t bm,
                                                     uint32_t bn,
                                                     uint32_t bk,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half_t *gA,
                                                     half_t *gB, half_t *gC)
{
    KPR_ASSERT(bm % 16U == 0U);
    KPR_ASSERT(bn % 16U == 0U);
    KPR_ASSERT(bk % 16U == 0U);
    KPR_GUARD(rows % bm == 0U);
    KPR_GUARD(shared % bk == 0U);
    KPR_GUARD(cols % bn == 0U);
    KPR_KCALL(__hoisted_21,
              rows / bm * (cols / bn),
              bm / 64U * (bn / 64U) * 32U,
              2U * (bm * bk) + 2U * (bk * bn),
              bm, bn, bk, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_8x8
*/
static void
__hoisted_22(uint32_t bm,
             uint32_t bn,
             uint32_t bk,
             uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U * (bm * bk));
    uint32_t num_k_tiles = shared / bk;
    uint32_t num_n_tiles = cols / bn;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t warpRow = threadIdx.x / 32U / (bn / 128U);
    uint32_t warpCol = threadIdx.x / 32U % (bn / 128U);
    auto &
        aFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_a,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        bFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
                                                           wmma::matrix_b,
                                                           16,
                                                           16,
                                                           16,
                                                           wmma::row_major),
                                         8U));
    auto &
        accFrags =
        KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE
                 (KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
                  64U));
    uint32_t fi = 0U;
    for (; fi < 64U; fi += 1U)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < bm * bk; i2 += bm / 128U * (bn / 128U) * 32U * 8U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i2 / bk;
            uint32_t col = i2 % bk;
            vec_memcpy(local,
                       tileA + shared * (mrow * bm) + __anf01 * bk +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sA[row * bk + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = threadIdx.x * 8U;
        for (; i < bk * bn; i += bm / 128U * (bn / 128U) * 32U * 8U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = i / bn;
            uint32_t col = i % bn;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * bk) + mcol * bn + cols * row +
                       col);
            uint32_t k = 0U;
            for (; k < 8U; k += 1U)
                sB[row * bn + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < bk / 16U; dotIdx += 1U) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U) {
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_a_tiles,
                                                  bk * (warpRow * 128U) +
                                                  __anf04 * 16U +
                                                  bk * (i0 * 16U)), bk);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U) {
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       kpr_offset(tile_for_tc_b_tiles,
                                                  bn * (__anf05 * 16U) +
                                                  warpCol * 128U + i1 * 16U),
                                       bn);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i += 1U) {
        uint32_t j = 0U;
        for (; j < 8U; j += 1U) {
            auto & __anf2 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(kpr_offset(gC,
                                               cols * (blockIdx.x /
                                                       (cols / bn) * bm) +
                                               blockIdx.x % (cols / bn) * bn +
                                               cols * (threadIdx.x / 32U /
                                                       (bn / 128U) * 128U)
                                               +
                                               threadIdx.x / 32U % (bn / 128U) *
                                               128U + cols * (i * 16U)
                                               + j * 16U), __anf2, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_8x8(uint32_t bm,
                                                     uint32_t bn,
                                                     uint32_t bk,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half_t *gA,
                                                     half_t *gB, half_t *gC)
{
    KPR_ASSERT(bm % 16U == 0U);
    KPR_ASSERT(bn % 16U == 0U);
    KPR_ASSERT(bk % 16U == 0U);
    KPR_GUARD(rows % bm == 0U);
    KPR_GUARD(shared % bk == 0U);
    KPR_GUARD(cols % bn == 0U);
    KPR_KCALL(__hoisted_22,
              rows / bm * (cols / bn),
              bm / 128U * (bn / 128U) * 32U,
              2U * (bm * bk) + 2U * (bk * bn),
              bm, bn, bk, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}
