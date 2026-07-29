
#include "Klas_GEMM_TensorCore2D_To.h"

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(4096U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2_0, nblk,
              128U, 8192U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(4096U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 32U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4_0, nblk, 64U,
              6144U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(4096U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2_0, nblk, 64U,
              6144U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(4096U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 64U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(5120U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 5120U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4_0, nblk, 32U,
              5120U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2_0, nblk,
              128U, 12288U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 32U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4_0, nblk, 64U,
              10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2_0, nblk, 64U,
              10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 64U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(9216U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 9216U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4_0, nblk, 32U,
              9216U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2_0, nblk,
              128U, 20480U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 32U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 18432U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4_0, nblk, 64U,
              18432U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 18432U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2_0, nblk, 64U,
              18432U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4_0(uint32_t shared,
                                                       uint32_t cols,
                                                       __nv_bfloat16 *gA,
                                                       __nv_bfloat16 *gB,
                                                       __nv_bfloat16 *gC,
                                                       __nv_bfloat16 *gD,
                                                       float alpha,
                                                       float beta,
                                                       uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 64U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4(uint32_t
                                                                     rows,
                                                                     uint32_t
                                                                     shared,
                                                                     uint32_t
                                                                     cols,
                                                                     __nv_bfloat16
                                                                     *gA,
                                                                     __nv_bfloat16
                                                                     *gB,
                                                                     __nv_bfloat16
                                                                     *gC,
                                                                     __nv_bfloat16
                                                                     *gD,
                                                                     float
                                                                     alpha,
                                                                     float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(17408U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 17408U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4_0, nblk, 32U,
              17408U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4_0, nblk,
              128U, 10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 32U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8_0, nblk,
              64U, 8192U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 4U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 4U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2_0, nblk,
              128U, 10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4_0, nblk,
              64U, 8192U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 1024U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 64U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(7168U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 7168U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8_0, nblk,
              32U, 7168U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 4U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 4U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2_0, nblk,
              256U, 20480U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4_0, nblk,
              128U, 16384U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 32U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 14336U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8_0, nblk,
              64U, 14336U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 4U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 4U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2_0, nblk,
              128U, 16384U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 14336U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4_0, nblk,
              64U, 14336U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 64U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(13312U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 13312U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8_0, nblk,
              32U, 13312U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 4U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 4U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2_0, nblk,
              256U, 32768U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 28672U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4_0, nblk,
              128U, 28672U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 32U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 26624U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8_0, nblk,
              64U, 26624U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 4U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 4U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 28672U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2_0, nblk,
              128U, 28672U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 26624U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4_0, nblk,
              64U, 26624U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 64U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 64U + threadIdx.x / 32U * 64U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(25600U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 25600U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8_0, nblk,
              32U, 25600U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 32U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4_0, nblk,
              128U, 10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2_0, nblk,
              128U, 10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 64U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4_0, nblk,
              64U, 8192U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 128U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2_0, nblk,
              64U, 8192U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 128U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(6144U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 128U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(7168U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 7168U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4_0, nblk,
              32U, 7168U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2_0, nblk,
              256U, 20480U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 32U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4_0, nblk,
              128U, 16384U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2_0, nblk,
              128U, 16384U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 64U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 14336U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4_0, nblk,
              64U, 14336U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 128U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 14336U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2_0, nblk,
              64U, 14336U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 128U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(12288U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 128U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(13312U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 13312U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4_0, nblk,
              32U, 13312U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2_0, nblk,
              256U, 32768U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 32U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 28672U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4_0, nblk,
              128U, 28672U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 28672U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2_0, nblk,
              128U, 28672U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 64U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 26624U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4_0, nblk,
              64U, 26624U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 32U +
                                        i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 128U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 64U + threadIdx.x / 32U % 2U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 26624U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2_0, nblk,
              64U, 26624U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4_0(uint32_t shared,
                                                        uint32_t cols,
                                                        __nv_bfloat16 *gA,
                                                        __nv_bfloat16 *gB,
                                                        __nv_bfloat16 *gC,
                                                        __nv_bfloat16 *gD,
                                                        float alpha,
                                                        float beta,
                                                        uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 64U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 128U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf02 * 16U + i1 * 16U), 64U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 64U);
        uint32_t mcol2 = blockIdx.x % (cols / 64U);
        float *sTile = (float *)KPR_SHMEM_AT(24576U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 128U + __anf02 / 4U * 16U +
                row;
            uint32_t globalCol = mcol2 * 64U + __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4(uint32_t
                                                                      rows,
                                                                      uint32_t
                                                                      shared,
                                                                      uint32_t
                                                                      cols,
                                                                      __nv_bfloat16
                                                                      *gA,
                                                                      __nv_bfloat16
                                                                      *gB,
                                                                      __nv_bfloat16
                                                                      *gC,
                                                                      __nv_bfloat16
                                                                      *gD,
                                                                      float
                                                                      alpha,
                                                                      float
                                                                      beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(25600U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 25600U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4_0, nblk,
              32U, 25600U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4_0, nblk,
              256U, 16384U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 32U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8_0, nblk,
              128U, 12288U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 4U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2_0, nblk,
              256U, 16384U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4_0, nblk,
              128U, 12288U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 64U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8_0, nblk,
              64U, 10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 4U) * 128U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 128U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2_0, nblk,
              128U, 12288U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 128U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4_0, nblk,
              64U, 10240U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = shared / 16U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 2048U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 16U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 16U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 16U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (16U * (threadIdx.x / 32U) * 128U +
                                        __anf01 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 64U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(8192U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 128U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(9216U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 9216U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8_0, nblk,
              32U, 9216U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 4096U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 4096U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 4U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2_0, nblk,
              512U, 32768U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 512U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0, nblk,
              256U, 24576U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 32U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8_0, nblk,
              128U, 20480U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 4U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2_0, nblk,
              256U, 24576U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4_0, nblk,
              128U, 20480U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 64U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 18432U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8_0, nblk,
              64U, 18432U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 4U) * 128U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 128U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2_0, nblk,
              128U, 20480U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 128U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 18432U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4_0, nblk,
              64U, 18432U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U) * 128U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 64U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 128U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(17408U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 17408U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8_0, nblk,
              32U, 17408U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 4096U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 4096U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 4U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 32U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2_0, nblk,
              512U, 49152U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 512U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4_0, nblk,
              256U, 40960U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 32U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 32U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(36864U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 36864U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8_0, nblk,
              128U, 36864U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 4U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 64U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2_0, nblk,
              256U, 40960U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 64U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(36864U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 36864U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4_0, nblk,
              128U, 36864U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 64U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 64U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(34816U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 34816U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8_0, nblk,
              64U, 34816U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 4U) * 128U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 2U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 4U * 32U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 4U * 128U +
                __anf02 / 2U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 4U * 32U +
                __anf02 % 2U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(36864U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 36864U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2_0, nblk,
              128U, 36864U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 128U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 128U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(34816U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 34816U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4_0, nblk,
              64U, 34816U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 64U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / 64U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 8U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 8192U; i2 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 64U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 64U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf0 * 64U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (64U * (threadIdx.x / 32U) * 128U +
                                        __anf01 * 16U + 64U * i0 * 16U), 64U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U + i1 * 16U), 128U);
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 64U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(32768U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / 16U;
            uint32_t col = __anf03 % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U * 128U + __anf02 / 8U * 16U +
                row;
            uint32_t globalCol = mcol2 * 128U + __anf02 % 8U * 16U + col;
            float av = sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(33792U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 33792U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8_0, nblk,
              32U, 33792U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 32U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
