
#include "Kuiper_GEMM_TensorCore2D.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_2x2
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_0, nblk, 128U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_2x4
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_1, nblk, 64U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x2
*/
static void __hoisted_2(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_2, nblk, 64U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x4
*/
static void __hoisted_3(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
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
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_3, nblk, 32U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_2x2
*/
static void __hoisted_4(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_4, nblk, 128U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_2x4
*/
static void __hoisted_5(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_5, nblk, 64U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x2
*/
static void __hoisted_6(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_6, nblk, 64U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x4
*/
static void __hoisted_7(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_7, nblk, 32U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x2
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
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
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_8, nblk, 128U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x4
*/
static void __hoisted_9(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_9, nblk, 64U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x2
*/
static void __hoisted_10(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_10, nblk, 64U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x4
*/
static void __hoisted_11(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
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
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_11, nblk, 32U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_2x4
*/
static void __hoisted_12(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_12, nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_2x8
*/
static void __hoisted_13(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_13, nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_4x2
*/
static void __hoisted_14(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_14, nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_4x4
*/
static void __hoisted_15(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_15, nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_4x8
*/
static void __hoisted_16(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_16, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_16, nblk, 32U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_2x2
*/
static void __hoisted_17(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_17, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_17, nblk, 256U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_2x4
*/
static void __hoisted_18(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_18, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_18, nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_2x8
*/
static void __hoisted_19(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_19, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_19, nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_4x2
*/
static void __hoisted_20(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_20, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_20, nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_4x4
*/
static void __hoisted_21(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_21, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_21, nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_4x8
*/
static void __hoisted_22(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_22, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_22, nblk, 32U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_2x2
*/
static void __hoisted_23(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_23, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_23, nblk, 256U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_2x4
*/
static void __hoisted_24(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_24, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_24, nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_2x8
*/
static void __hoisted_25(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_25, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_25, nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_4x2
*/
static void __hoisted_26(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_26, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_26, nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_4x4
*/
static void __hoisted_27(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_27, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_27, nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_4x8
*/
static void __hoisted_28(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_28, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_28, nblk, 32U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_2x4
*/
static void __hoisted_29(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_29, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_29, nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_4x2
*/
static void __hoisted_30(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_30, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_30, nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_4x4
*/
static void __hoisted_31(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_31, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_31, nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_8x2
*/
static void __hoisted_32(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_32, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_32, nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_8x4
*/
static void __hoisted_33(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 128U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_8x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_33, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_33, nblk, 32U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_2x2
*/
static void __hoisted_34(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_34, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_34, nblk, 256U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_2x4
*/
static void __hoisted_35(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_35, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_35, nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_4x2
*/
static void __hoisted_36(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_36, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_36, nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_4x4
*/
static void __hoisted_37(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_37, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_37, nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_8x2
*/
static void __hoisted_38(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_38, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_38, nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_8x4
*/
static void __hoisted_39(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 128U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_8x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_39, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_39, nblk, 32U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_2x2
*/
static void __hoisted_40(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_40, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_40, nblk, 256U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_2x4
*/
static void __hoisted_41(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_41, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_41, nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_4x2
*/
static void __hoisted_42(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_42, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_42, nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_4x4
*/
static void __hoisted_43(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_43, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_43, nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_8x2
*/
static void __hoisted_44(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U +
                                       __anf07 * 16U, 64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_44, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_44, nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_8x4
*/
static void __hoisted_45(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 128U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf06 * 16U) + __anf07 * 16U,
                                       64U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_8x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t *gA,
                                                               half_t *gB,
                                                               half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_45, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_45, nblk, 32U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_2x4
*/
static void __hoisted_46(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_2x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_46, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_46, nblk, 256U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_2x8
*/
static void __hoisted_47(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_2x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_47, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_47, nblk, 128U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_4x2
*/
static void __hoisted_48(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_48, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_48, nblk, 256U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_4x4
*/
static void __hoisted_49(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_49, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_49, nblk, 128U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_4x8
*/
static void __hoisted_50(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_50, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_50, nblk, 64U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_8x2
*/
static void __hoisted_51(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 4U * 128U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 128U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_51, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_51, nblk, 128U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_8x4
*/
static void __hoisted_52(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_52, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_52, nblk, 64U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_8x8
*/
static void __hoisted_53(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 128U) +
                                       __anf05 * 16U + 16U * (__anf06 * 16U),
                                       16U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_53, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_53, nblk, 32U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x2
*/
static void __hoisted_54(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_54, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_54, nblk, 512U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x4
*/
static void __hoisted_55(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_55, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_55, nblk, 256U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x8
*/
static void __hoisted_56(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_56, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_56, nblk, 128U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x2
*/
static void __hoisted_57(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_57, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_57, nblk, 256U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x4
*/
static void __hoisted_58(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_58, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_58, nblk, 128U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x8
*/
static void __hoisted_59(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_59, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_59, nblk, 64U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x2
*/
static void __hoisted_60(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 128U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 128U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_60, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_60, nblk, 128U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x4
*/
static void __hoisted_61(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_61, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_61, nblk, 64U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x8
*/
static void __hoisted_62(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 128U) +
                                       __anf05 * 16U + 32U * (__anf06 * 16U),
                                       32U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_62, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_62, nblk, 32U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x2
*/
static void __hoisted_63(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 4096U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_63, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_63, nblk, 512U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x4
*/
static void __hoisted_64(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_64, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_64, nblk, 256U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x8
*/
static void __hoisted_65(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 2U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_65, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_65, nblk, 128U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x2
*/
static void __hoisted_66(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_66, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_66, nblk, 256U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x4
*/
static void __hoisted_67(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_67, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_67, nblk, 128U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x8
*/
static void __hoisted_68(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_68, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_68, nblk, 64U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x2
*/
static void __hoisted_69(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 128U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 2U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 2U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 128U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_69, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_69, nblk, 128U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x4
*/
static void __hoisted_70(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf07 * 16U, 128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf1 * 16U)
                                    + __anf0 * 16U, __anf02, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_70, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_70, nblk, 64U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x8
*/
static void __hoisted_71(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
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
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            half_t local[8U];
            memset(local, 0U, 8U * sizeof(half_t));
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf06 = i0;
                auto & __anf11 = aFrags[i0];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 128U) +
                                       __anf05 * 16U + 64U * (__anf06 * 16U),
                                       64U);
            }
            uint32_t __anf06 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf07 = i1;
                auto & __anf11 = bFrags[i1];
                wmma::load_matrix_sync(__anf11,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf06 * 16U) + __anf07 * 16U,
                                       128U);
            }
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf1 = i;
            uint32_t __anf0 = j;
            auto & __anf02 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (__anf1 * 16U)
                                    + __anf0 * 16U,
                                    __anf02, cols, wmma::mem_row_major);
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
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_71, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_71, nblk, 32U, 32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
