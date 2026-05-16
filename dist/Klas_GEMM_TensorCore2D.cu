
#include "Klas_GEMM_TensorCore2D.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_2x2_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_2x2(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_2x2_0,
              nblk, 128U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_2x4_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_2x4(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_2x4_0,
              nblk, 64U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_4x2_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x2(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_4x2_0,
              nblk, 64U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_4x4_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x4(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_4x4_0,
              nblk, 32U, 4096U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_2x2_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_2x2(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_2x2_0,
              nblk, 128U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_2x4_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_2x4(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_2x4_0,
              nblk, 64U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_4x2_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x2(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_4x2_0,
              nblk, 64U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_4x4_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x32_16x16x16_4x4_0,
              nblk, 32U, 8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_2x2_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x2(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_2x2_0,
              nblk, 128U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_2x4_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x4(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_2x4_0,
              nblk, 64U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_4x2_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x2(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_4x2_0,
              nblk, 64U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_4x4_0(uint32_t shared,
                                                 uint32_t cols,
                                                 half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x4(uint32_t rows,
                                                            uint32_t shared,
                                                            uint32_t cols,
                                                            half *gA,
                                                            half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_4x4_0,
              nblk, 32U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_2x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_2x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_2x4_0,
              nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_2x8
*/
static void
__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_2x8_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_2x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_2x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_2x8_0,
              nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x2_0,
              nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x4_0,
              nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x16_16x16x16_4x8
*/
static void
__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x8_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x16_16x16x16_4x8_0,
              nblk, 32U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x2_0,
              nblk, 256U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x4_0,
              nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_2x8
*/
static void
__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x8_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_2x8_0,
              nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x2_0,
              nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x4_0,
              nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x32_16x16x16_4x8
*/
static void
__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x8_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x32_16x16x16_4x8_0,
              nblk, 32U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x2_0,
              nblk, 256U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x4_0,
              nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_2x8
*/
static void
__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x8_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_2x8_0,
              nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x2_0,
              nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x4_0,
              nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x128x64_16x16x16_4x8
*/
static void
__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x8_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 64U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x8(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x128x64_16x16x16_4x8_0,
              nblk, 32U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_2x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_2x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_2x4_0,
              nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_4x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_4x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_4x2_0,
              nblk, 128U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_4x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_4x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_4x4_0,
              nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_8x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_8x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_8x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_8x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_8x2_0,
              nblk, 64U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x16_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_8x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 128U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_8x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_8x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x16_16x16x16_8x4_0,
              nblk, 32U, 6144U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_2x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_2x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_2x2_0,
              nblk, 256U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_2x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_2x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_2x4_0,
              nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_4x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_4x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_4x2_0,
              nblk, 128U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_4x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_4x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_4x4_0,
              nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_8x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_8x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_8x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_8x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_8x2_0,
              nblk, 64U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_8x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 128U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_8x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_8x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x32_16x16x16_8x4_0,
              nblk, 32U, 12288U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_2x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_2x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_2x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_2x2_0,
              nblk, 256U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_2x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_2x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_2x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_2x4_0,
              nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_4x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_4x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_4x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_4x2_0,
              nblk, 128U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_4x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_4x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_4x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_4x4_0,
              nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_8x2
*/
static void
__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_8x2_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 32U + i1 * 16U,
                                       64U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_8x2(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_8x2_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_8x2_0,
              nblk, 64U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x64x64_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_8x4_0(uint32_t shared,
                                                  uint32_t cols,
                                                  half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 128U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       64U * (__anf011 * 16U) + i1 * 16U, 64U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 128U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 4U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_8x4(uint32_t rows,
                                                             uint32_t shared,
                                                             uint32_t cols,
                                                             half *gA,
                                                             half *gB, half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_8x4_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x64x64_16x16x16_8x4_0,
              nblk, 32U, 24576U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_2x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_2x4_0, nblk, 256U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_2x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_2x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_2x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_2x8_0, nblk, 128U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x2_0, nblk, 256U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x4_0, nblk, 128U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_4x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_4x8_0, nblk, 64U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_8x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 4U * 128U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 128U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x2_0, nblk, 128U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x4_0, nblk, 64U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x16_16x16x16_8x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(4096U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 16U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 16U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       16U * (threadIdx.x / 32U * 128U) +
                                       __anf010 * 16U + 16U * (i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x16_16x16x16_8x8_0, nblk, 32U,
              8192U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 4096U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 4096U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x2_0, nblk, 512U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x4_0, nblk, 256U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_2x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_2x8_0, nblk, 128U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x2_0, nblk, 256U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x4_0, nblk, 128U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_4x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_4x8_0, nblk, 64U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 4U * 128U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 128U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x2_0, nblk, 128U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x4_0, nblk, 64U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 128U) +
                                       __anf010 * 16U + 32U * (i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x8_0, nblk, 32U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 4096U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 4096U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 32U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x2_0, nblk, 512U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 32U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x4_0, nblk, 256U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_2x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 32U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 32U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_2x8_0, nblk, 128U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 64U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x2_0, nblk, 256U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 64U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x4_0, nblk, 128U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_4x8_0, nblk, 64U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x2
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x2_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 4U * 128U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 4U * 32U + i1 * 16U,
                                       128U);
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
        for (; j < 2U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 4U * 128U)
                                    + threadIdx.x / 32U % 4U * 32U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 2U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x2_0, nblk, 128U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U + i1 * 16U,
                                       128U);
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
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (i * 16U)
                                    + j * 16U, accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x4_0, nblk, 64U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_8x8
*/
static void
__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x8_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(16384U);
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
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf03 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 128U) +
                                       __anf010 * 16U + 64U * (i0 * 16U), 64U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       128U * (__anf011 * 16U) + i1 * 16U,
                                       128U);
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
        for (; j < 8U; j++)
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 128U)
                                    + cols * (i * 16U)
                                    + j * 16U,
                                    accFrags[i * 8U + j],
                                    cols, wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x8(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
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
         (__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x64_16x16x16_8x8_0, nblk, 32U,
              32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
