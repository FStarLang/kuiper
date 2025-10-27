
#include "Kuiper_GEMM_TensorCorePiped.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x4
*/
static void __hoisted_7(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    // USE TWICE AS MUCH SPACE TO STORE TILES
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U*4096U);
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
    
    // ALLOCATE REGISTER BUFFERS
    half_t reg_bufA[(2048U/256U) * 8U];
    memset(reg_bufA, 0U, (2048U/256U) * 8U * sizeof(half_t));
    half_t reg_bufB[(2048U/256U) * 8U];
    memset(reg_bufB, 0U, (2048U/256U) * 8U * sizeof(half_t));
   
    // SUPERFLUOUS BARRIER FOR WRITE PERMISSION TO SHARED MEMORY IN KUIPER
    __syncthreads();
    // BEGIN LOAD FIRST TILES
    half_t *tileA = gA;
    uint32_t i2 = 0U;
    for (; i2 < 2048U; i2 += 256U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
        uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileA + shared * (mrow * 64U) + 0 * 32U +
                   shared * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sA[row * 32U + col + k] = local[k];
    }
    half_t *tileB = gB;
    for (uint32_t i = 0U; i < 2048U; i += 256U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i + threadIdx.x * 8U) / 64U;
        uint32_t col = (i + threadIdx.x * 8U) % 64U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileB + cols * (0 * 32U) + mcol * 64U +
                   cols * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sB[row * 64U + col + k] = local[k];
    }
    
    // GPUGenie SYNCs HERE
    
    // KEEP TRACK OF PIPELINE STAGE
    uint32_t stage = 0;
    // SET bkIdx = 1
    uint32_t bkIdx = 1U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
    // KEEP SYNCHRONIZING AT BEGINNING OF LOOP
        __syncthreads();
    // FLIP-FLOP NEXT PIPELINE STAGE
        uint32_t next_stage = stage ^ 1;

        uint32_t __anf01 = bkIdx;
    // BEGIN LOAD NEXT TILE INTO REGISTERS
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

    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufA[(i2/256) * 8U + k] = local[k];

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
    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufB[(i/256U) * 8U + k] = local[k];
        }
    // END LOAD NEXT TILE INTO REGISTERS

        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf04 = dotIdx;
    // OFFSET SHARED MEMORY DEPENDING ON STAGE
            half_t *tile_for_tc_a_tiles = sA + stage * 2048U;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf05 = i0;
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf04 * 16U + 32U * (__anf05 * 16U),
                                       32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB + stage * 2048U;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf06 = i1;
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf05 * 16U) + __anf06 * 16U,
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
        
        // BEGIN STORE FROM REGISTERS INTO SHARED MEMORY
        uint32_t i3 = 0U;
        for (; i3 < 2048U; i3 += 256U) {
            uint32_t row = (i3 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i3 + threadIdx.x * 8U) % 32U;

            for (uint32_t k = 0; k < 8U; k++) {
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sA[next_stage * 2048U + row * 32U + col + k] = reg_bufA[(i3/256U) * 8U + k];
            }
        }
        for (uint32_t i4 = 0U; i4 < 2048U; i4 += 256U) {
            uint32_t row = (i4 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i4 + threadIdx.x * 8U) % 64U;

            for (uint32_t k = 0U; k < 8U; k++)
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sB[next_stage * 2048U + row * 64U + col + k] = reg_bufB[(i4/256U) * 8U + k];
        }
        // END STORE FROM REGISTERS INTO SHARED MEMORY
        
        stage ^= 1;
    }
    
    // BEGIN PROCESS LAST TILE
    // SYNCHRONIZE FOR READ PERMISSION ON NEW STAGE
    __syncthreads();
    for (uint32_t dotIdx = 0U; dotIdx < 2U; dotIdx++) {
        uint32_t __anf04 = dotIdx;
        half_t *tile_for_tc_a_tiles = sA + stage * 2048U;
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++) {
            uint32_t __anf05 = i0;
            auto & __anf1 = aFrags[i0];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_a_tiles +
                                   32U * (threadIdx.x / 32U * 64U) +
                                   __anf04 * 16U + 32U * (__anf05 * 16U),
                                   32U);
        }
        uint32_t __anf05 = dotIdx;
        half_t *tile_for_tc_b_tiles = sB + stage * 2048U;
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++) {
            uint32_t __anf06 = i1;
            auto & __anf1 = bFrags[i1];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_b_tiles +
                                   64U * (__anf05 * 16U) + __anf06 * 16U,
                                   64U);
        }
        uint32_t resIdxM = 0U;
        for (; resIdxM < 4U; resIdxM++) {
            uint32_t resIdxN = 0U;
            for (; resIdxN < 4U; resIdxN++) {
                auto& acc_frag = accFrags[resIdxM * 4U + resIdxN];
                wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                               acc_frag);
            }
        }
    }
    // END PROCESS LAST TILE

    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf11 = i;
            uint32_t __anf01 = j;
            auto & __anf03 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf11 * 16U)
                                    + __anf01 * 16U,
                                    __anf03, cols, wmma::mem_row_major);
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
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    // DOUBLE AMOUNT OF SHARED MEMORY
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_7, nblk, 32U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x4
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    // USE TWICE AS MUCH SPACE TO STORE TILES
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U*4096U);
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
    
    // ALLOCATE REGISTER BUFFERS
    half_t reg_bufA[(2048U/256U) * 8U];
    memset(reg_bufA, 0U, (2048U/256U) * 8U * sizeof(half_t));
    half_t reg_bufB[(2048U/256U) * 8U];
    memset(reg_bufB, 0U, (2048U/256U) * 8U * sizeof(half_t));
   
    // BEGIN LOAD FIRST TILES
    half_t *tileA = gA;
    uint32_t i2 = 0U;
    for (; i2 < 2048U; i2 += 256U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
        uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileA + shared * (mrow * 64U) + 0 * 32U +
                   shared * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sA[row * 32U + col + k] = local[k];
    }
    half_t *tileB = gB;
    for (uint32_t i = 0U; i < 2048U; i += 256U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i + threadIdx.x * 8U) / 64U;
        uint32_t col = (i + threadIdx.x * 8U) % 64U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileB + cols * (0 * 32U) + mcol * 64U +
                   cols * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sB[row * 64U + col + k] = local[k];
    }
    
    // GPUGenie SYNCs HERE
    __syncthreads();
    
    // KEEP TRACK OF PIPELINE STAGE
    uint32_t stage = 0;
    // SET bkIdx = 1
    uint32_t bkIdx = 1U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
    // FLIP-FLOP NEXT PIPELINE STAGE
        uint32_t next_stage = stage ^ 1;

        uint32_t __anf01 = bkIdx;
    // BEGIN LOAD NEXT TILE INTO REGISTERS
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

    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufA[(i2/256) * 8U + k] = local[k];

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
    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufB[(i/256U) * 8U + k] = local[k];
        }
    // END LOAD NEXT TILE INTO REGISTERS

        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf04 = dotIdx;
    // OFFSET SHARED MEMORY DEPENDING ON STAGE
            half_t *tile_for_tc_a_tiles = sA + stage * 2048U;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf05 = i0;
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf04 * 16U + 32U * (__anf05 * 16U),
                                       32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB + stage * 2048U;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf06 = i1;
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf05 * 16U) + __anf06 * 16U,
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
        
        // BEGIN STORE FROM REGISTERS INTO SHARED MEMORY
        uint32_t i3 = 0U;
        for (; i3 < 2048U; i3 += 256U) {
            uint32_t row = (i3 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i3 + threadIdx.x * 8U) % 32U;

            for (uint32_t k = 0; k < 8U; k++) {
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sA[next_stage * 2048U + row * 32U + col + k] = reg_bufA[(i3/256U) * 8U + k];
            }
        }
        for (uint32_t i4 = 0U; i4 < 2048U; i4 += 256U) {
            uint32_t row = (i4 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i4 + threadIdx.x * 8U) % 64U;

            for (uint32_t k = 0U; k < 8U; k++)
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sB[next_stage * 2048U + row * 64U + col + k] = reg_bufB[(i4/256U) * 8U + k];
        }
        // END STORE FROM REGISTERS INTO SHARED MEMORY
        
        stage ^= 1;
        __syncthreads();
    }
    
    // BEGIN PROCESS LAST TILE
    for (uint32_t dotIdx = 0U; dotIdx < 2U; dotIdx++) {
        uint32_t __anf04 = dotIdx;
        half_t *tile_for_tc_a_tiles = sA + stage * 2048U;
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++) {
            uint32_t __anf05 = i0;
            auto & __anf1 = aFrags[i0];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_a_tiles +
                                   32U * (threadIdx.x / 32U * 64U) +
                                   __anf04 * 16U + 32U * (__anf05 * 16U),
                                   32U);
        }
        uint32_t __anf05 = dotIdx;
        half_t *tile_for_tc_b_tiles = sB + stage * 2048U;
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++) {
            uint32_t __anf06 = i1;
            auto & __anf1 = bFrags[i1];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_b_tiles +
                                   64U * (__anf05 * 16U) + __anf06 * 16U,
                                   64U);
        }
        uint32_t resIdxM = 0U;
        for (; resIdxM < 4U; resIdxM++) {
            uint32_t resIdxN = 0U;
            for (; resIdxN < 4U; resIdxN++) {
                auto& acc_frag = accFrags[resIdxM * 4U + resIdxN];
                wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                               acc_frag);
            }
        }
    }
    // END PROCESS LAST TILE

    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf11 = i;
            uint32_t __anf01 = j;
            auto & __anf03 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf11 * 16U)
                                    + __anf01 * 16U,
                                    __anf03, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCorePiped_change_sync_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
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
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    // DOUBLE AMOUNT OF SHARED MEMORY
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_0, nblk, 32U, 16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x32_16x16x16_4x4
*/
static void __hoisted_8(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB,
                        half_t *gC)
{
    half_t *sA = (half_t *) KPR_SHMEM_AT(0U);
    // USE TWICE AS MUCH SPACE TO STORE TILES
    half_t *sB = (half_t *) KPR_SHMEM_AT(2U*4096U);
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
    
    // ALLOCATE REGISTER BUFFERS
    // GPUGenie uses alignas(16), not sure why
    alignas(16) half_t reg_bufA[(2048U/256U) * 8U];
    memset(reg_bufA, 0U, (2048U/256U) * 8U * sizeof(half_t));
    alignas(16) half_t reg_bufB[(2048U/256U) * 8U];
    memset(reg_bufB, 0U, (2048U/256U) * 8U * sizeof(half_t));
   
    // BEGIN LOAD FIRST TILES
    half_t *tileA = gA;
    uint32_t i2 = 0U;
    for (; i2 < 2048U; i2 += 256U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
        uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileA + shared * (mrow * 64U) + 0 * 32U +
                   shared * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sA[row * 32U + col + k] = local[k];
    }
    half_t *tileB = gB;
    for (uint32_t i = 0U; i < 2048U; i += 256U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i + threadIdx.x * 8U) / 64U;
        uint32_t col = (i + threadIdx.x * 8U) % 64U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileB + cols * (0 * 32U) + mcol * 64U +
                   cols * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sB[row * 64U + col + k] = local[k];
    }
    
    // GPUGenie SYNCs HERE
    __syncthreads();
    
    // KEEP TRACK OF PIPELINE STAGE
    uint32_t stage = 0;
    // SET bkIdx = 1
    uint32_t bkIdx = 1U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
    // FLIP-FLOP NEXT PIPELINE STAGE
        uint32_t next_stage = stage ^ 1;

        uint32_t __anf01 = bkIdx;
    // BEGIN LOAD NEXT TILE INTO REGISTERS
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

    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufA[(i2/256) * 8U + k] = local[k];

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
    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufB[(i/256U) * 8U + k] = local[k];
        }
    // END LOAD NEXT TILE INTO REGISTERS

        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf04 = dotIdx;
    // OFFSET SHARED MEMORY DEPENDING ON STAGE
            half_t *tile_for_tc_a_tiles = sA + stage * 2048U;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++) {
                uint32_t __anf05 = i0;
                auto & __anf1 = aFrags[i0];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_a_tiles +
                                       32U * (threadIdx.x / 32U * 64U) +
                                       __anf04 * 16U + 32U * (__anf05 * 16U),
                                       32U);
            }
            uint32_t __anf05 = dotIdx;
            half_t *tile_for_tc_b_tiles = sB + stage * 2048U;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++) {
                uint32_t __anf06 = i1;
                auto & __anf1 = bFrags[i1];
                wmma::load_matrix_sync(__anf1,
                                       tile_for_tc_b_tiles +
                                       64U * (__anf05 * 16U) + __anf06 * 16U,
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
        
        // BEGIN STORE FROM REGISTERS INTO SHARED MEMORY
        uint32_t i3 = 0U;
        for (; i3 < 2048U; i3 += 256U) {
            uint32_t row = (i3 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i3 + threadIdx.x * 8U) % 32U;

            for (uint32_t k = 0; k < 8U; k++) {
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sA[next_stage * 2048U + row * 32U + col + k] = reg_bufA[(i3/256U) * 8U + k];
            }
        }
        for (uint32_t i4 = 0U; i4 < 2048U; i4 += 256U) {
            uint32_t row = (i4 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i4 + threadIdx.x * 8U) % 64U;

            for (uint32_t k = 0U; k < 8U; k++)
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sB[next_stage * 2048U + row * 64U + col + k] = reg_bufB[(i4/256U) * 8U + k];
        }
        // END STORE FROM REGISTERS INTO SHARED MEMORY
        
        stage ^= 1;
        __syncthreads();
    }
    
    // BEGIN PROCESS LAST TILE
    for (uint32_t dotIdx = 0U; dotIdx < 2U; dotIdx++) {
        uint32_t __anf04 = dotIdx;
        half_t *tile_for_tc_a_tiles = sA + stage * 2048U;
        uint32_t i0 = 0U;
        for (; i0 < 4U; i0++) {
            uint32_t __anf05 = i0;
            auto & __anf1 = aFrags[i0];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_a_tiles +
                                   32U * (threadIdx.x / 32U * 64U) +
                                   __anf04 * 16U + 32U * (__anf05 * 16U),
                                   32U);
        }
        uint32_t __anf05 = dotIdx;
        half_t *tile_for_tc_b_tiles = sB + stage * 2048U;
        uint32_t i1 = 0U;
        for (; i1 < 4U; i1++) {
            uint32_t __anf06 = i1;
            auto & __anf1 = bFrags[i1];
            wmma::load_matrix_sync(__anf1,
                                   tile_for_tc_b_tiles +
                                   64U * (__anf05 * 16U) + __anf06 * 16U,
                                   64U);
        }
        uint32_t resIdxM = 0U;
        for (; resIdxM < 4U; resIdxM++) {
            uint32_t resIdxN = 0U;
            for (; resIdxN < 4U; resIdxN++) {
                auto& acc_frag = accFrags[resIdxM * 4U + resIdxN];
                wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                               acc_frag);
            }
        }
    }
    // END PROCESS LAST TILE

    uint32_t i = 0U;
    for (; i < 4U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++) {
            uint32_t __anf11 = i;
            uint32_t __anf01 = j;
            auto & __anf03 = accFrags[i * 4U + j];
            wmma::store_matrix_sync(gC +
                                    cols * (blockIdx.x / (cols / 64U) * 64U) +
                                    blockIdx.x % (cols / 64U) * 64U +
                                    cols * (threadIdx.x / 32U * 64U)
                                    + cols * (__anf11 * 16U)
                                    + __anf01 * 16U,
                                    __anf03, cols, wmma::mem_row_major);
        }
    }
}

void
Kuiper_GEMM_TensorCorePiped_align_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
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
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    // DOUBLE AMOUNT OF SHARED MEMORY
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_8, nblk, 32U, 16384U, shared, cols, gA, gB, gC);
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
    
    // ALLOCATE REGISTER BUFFERS
    // GPUGenie uses alignas(16), not sure why
    alignas(16) half_t reg_bufA[(8192U/512U) * 8U];
    memset(reg_bufA, 0U, (8192U/512U) * 8U * sizeof(half_t));
    alignas(16) half_t reg_bufB[(8192U/512U) * 8U];
    memset(reg_bufB, 0U, (8192U/512U) * 8U * sizeof(half_t));

    // BEGIN LOAD FIRST TILES
    half_t *tileA = gA;
    uint32_t i2 = 0U;
    for (; i2 < 8192U; i2 += 512U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
        uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileA + shared * (mrow * 128U) + 0 * 64U +
                   shared * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sA[row * 64U + col + k] = local[k];
    }
    half_t *tileB = gB;
    for (uint32_t i = 0U; i < 8192U; i += 512U) {
        half_t local[8U];
        memset(local, 0U, 8U * sizeof(half_t));
        uint32_t row = (i + threadIdx.x * 8U) / 128U;
        uint32_t col = (i + threadIdx.x * 8U) % 128U;
        vec_memcpy(local,
    // SET __anf01 == 0
                   tileB + cols * (0 * 64U) + mcol * 128U +
                   cols * row + col);
        uint32_t k = 0U;
        for (; k < 8U; k++)
            sB[row * 128U + col + k] = local[k];
    }
    
    // GPUGenie SYNCs HERE
    __syncthreads();
    // END LOAD FIRST TILES
    
    // KEEP TRACK OF PIPELINE STAGE
    uint32_t stage = 0U;
    // SET bkIdx = 1
    uint32_t bkIdx = 1U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
    // FLIP-FLOP NEXT PIPELINE STAGE
        uint32_t next_stage = stage ^ 1;

        uint32_t __anf01 = bkIdx;
    // BEGIN LOAD NEXT TILE INTO REGISTERS
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

    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufA[(i2/512) * 8U + k] = local[k];
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

    // SUPERFLUOUS COPY FROM VECTOR REGISTERS INTO REGISTER BUFFER
            uint32_t k = 0U;
            for (; k < 8U; k++)
                reg_bufB[(i2/512) * 8U + k] = local[k];
        }
    // END LOAD NEXT TILE INTO REGISTERS
    
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf04 = dotIdx;
    // OFFSET SHARED MEMORY DEPENDING ON STAGE
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

        // BEGIN STORE FROM REGISTERS INTO SHARED MEMORY
        uint32_t i3 = 0U;
        for (; i3 < 8192U; i3 += 512U) {
            uint32_t row = (i3 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i3 + threadIdx.x * 8U) % 64U;

            for (uint32_t k = 0; k < 8U; k++) {
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sA[next_stage * 8192U + row * 64U + col + k] = reg_bufA[(i3/512U) * 8U + k];
            }
        }
        for (uint32_t i4 = 0U; i4 < 8192U; i4 += 512U) {
            uint32_t row = (i4 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i4 + threadIdx.x * 8U) % 128U;

            for (uint32_t k = 0U; k < 8U; k++)
        // OFFSET SHARED MEMORY DEPENDING ON STAGE
                sB[next_stage * 8192U + row * 128U + col + k] = reg_bufB[(i4/512U) * 8U + k];
        }
        // END STORE FROM REGISTERS INTO SHARED MEMORY
        
        stage ^= 1;
        __syncthreads();

    }
    
    // BEGIN PROCESS LAST TILE
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
Kuiper_GEMM_TensorCorePiped_align_g_gemm_f16_f16_128x128x64_16x16x16_4x8(uint32_t rows,
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