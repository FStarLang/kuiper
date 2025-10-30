#include <cuda/barrier>
#include "Kuiper_GEMM_TensorCoreAsync.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x4
*/
static void __hoisted_61(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    half_t *sB = (half_t *) KPR_SHMEM_AT(8192U);

    KPR_INIT_BARRIER(64U);

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
        // GPUGenie uses barrier.arrive_and_wait() for this synchronization,
        //  but the CUDA documentation's example does not
        __syncthreads();


        // VECTORIZED ASYNCHRONOUS COPIES BY THREAD
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy_async(
                &sA[row * 32U + col],
                &tileA[shared * (mrow * 128U) + __anf01 * 32U + shared * row + col],
                KPR_BARRIER
            );
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy_async(
                &sB[row * 128U + col],
                &tileB[cols * (__anf01 * 32U) + mcol * 128U + cols * row + col],
                KPR_BARRIER
            );
        }
        // WAIT FOR COPIES TO FINISH
        KPR_BARRIER.arrive_and_wait();

        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++) {
                uint32_t __anf05 = i0;
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U / 2U * 128U) +
                                       __anf04 * 16U + 32U * (__anf05 * 16U),
                                       32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf06 = i1;
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf05 * 16U) +
                                       threadIdx.x / 32U % 2U * 64U +
                                       __anf06 * 16U, 128U);
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
            uint32_t __anf11 = i;
            uint32_t __anf01 = j;
            auto & __anf03 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U / 2U * 128U)
                                    + threadIdx.x / 32U % 2U * 64U +
                                    cols * (__anf11 * 16U)
                                    + __anf01 * 16U, __anf03, cols,
                                    wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCoreAsync_g_gemm_f16_f16_128x128x32_16x16x16_8x4(uint32_t rows,
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
  hoisted when extracting g_gemm_f16_f16_128x128x64_16x16x16_4x8
*/
static void __hoisted_68(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                         half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    // USE TWICE AS MUCH SPACE TO STORE TILES
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U*16384U);

    // ALLOCATE AND INITIALIZE IN SHARED MEMORY
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> __barrier;
    if (threadIdx.x == 0) {
    // INITIALIZE WITH NUMBER OF THREADS PER BLOCK
        init(&__barrier, 64U);
    }
    __syncthreads();

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
    
    // BEGIN LOAD FIRST TILES
    half_t *tileA = gA;
    uint32_t i2 = 0U;
    for (; i2 < 8192U; i2 += 512U) {
        uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
        uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
        memcpy_async(
            &sA[row * 64U + col], 
            &tileA[shared * (mrow * 128U) + 0 * 64U + shared * row + col],
            cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
            __barrier
        );
    }
    half_t *tileB = gB;
    for (uint32_t i = 0U; i < 8192U; i += 512U) {
        uint32_t row = (i + threadIdx.x * 8U) / 128U;
        uint32_t col = (i + threadIdx.x * 8U) % 128U;
        memcpy_async(
            &sB[row * 128U + col],
            &tileB[cols * (0 * 64U) + mcol * 128U + cols * row + col],
            cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
            __barrier
        );
    }
    // END LOAD FIRST TILES
    
    uint32_t stage = 0U;
    uint32_t bkIdx = 1U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        // WAIT FOR COPIES TO FINISH
        __barrier.arrive_and_wait();

        uint32_t next_stage = stage ^ 1;

        // VECTORIZED ASYNCHRONOUS COPIES BY THREAD
        uint32_t __anf01 = bkIdx;
        half_t *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            memcpy_async(
                &sA[next_stage * 8192U + row * 64U + col], 
                &tileA[shared * (mrow * 128U) + __anf01 * 64U + shared * row + col],
                cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                __barrier
            );
        }
        half_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            memcpy_async(
                &sB[next_stage * 8192U + row * 128U + col],
                &tileB[cols * (__anf01 * 64U) + mcol * 128U + cols * row + col],
                cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                __barrier
            );
        }

        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {

            uint32_t __anf04 = dotIdx;
            half_t *tile_for_tc_a_tiles = sA + stage * 8192U;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf05 = i0;
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_a_tiles +
                                       64U * (threadIdx.x / 32U * 64U) +
                                       __anf04 * 16U + 64U * (__anf05 * 16U),
                                       64U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB + stage * 8192U;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++) {
                uint32_t __anf06 = i1;
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_b_tiles +
                                       128U * (__anf05 * 16U) + __anf06 * 16U,
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
        
        stage ^= 1;
    }
    
    // BEGIN PROCESS LAST TILE
    // WAIT FOR COPIES TO FINISH
    __barrier.arrive_and_wait();
    for (uint32_t dotIdx = 0U; dotIdx < 4U; dotIdx++) {
        uint32_t __anf04 = dotIdx;
        half_t *tile_for_tc_a_tiles = sA + stage * 8192U;
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++) {
            uint32_t __anf05 = i0;
            auto & __anf1 = aFrags[i0];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_a_tiles +
                                   64U * (threadIdx.x / 32U * 64U) +
                                   __anf04 * 16U + 64U * (__anf05 * 16U),
                                   64U);
        }
        uint32_t __anf05 = dotIdx;
        half_t *tile_for_tc_b_tiles = sB + stage * 8192U;
        uint32_t i1 = 0U;
        for (; i1 < 8U; i1++) {
            uint32_t __anf06 = i1;
            auto & __anf1 = bFrags[i1];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_b_tiles +
                                   128U * (__anf05 * 16U) + __anf06 * 16U,
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
    // END PROCESS LAST TILE

    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 8U; j++) {
            uint32_t __anf11 = i;
            uint32_t __anf01 = j;
            auto & __anf03 = accFrags[i * 8U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 128U) * 128U) +
                                    blockIdx.x % (cols / 128U) * 128U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf11 * 16U)
                                    + __anf01 * 16U,
                                    __anf03, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCorePipedAsync_g_gemm_f16_f16_128x128x64_16x16x16_4x8(uint32_t rows,
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
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    // DOUBLE AMOUNT OF SHARED MEMORY
    KPR_SHMEM_FITS(2U*32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_68, cudaFuncAttributeMaxDynamicSharedMemorySize, 2U*32768U));
    KPR_KCALL(__hoisted_68, nblk, 64U, 2U*32768U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}