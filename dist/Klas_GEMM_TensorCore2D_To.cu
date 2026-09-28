
#include "Klas_GEMM_TensorCore2D_To.h"

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 32U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 4U) * 64U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 4U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 64U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 4U) * 32U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 4U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 32U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 4U) * 64U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 4U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 64U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 4U) * 32U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 4U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 32U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 4U) * 64U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 4U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 64U + threadIdx.x / 32U * 64U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 128U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 128U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(6144U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 128U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 128U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 128U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 128U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(12288U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 128U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 128U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 128U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U +
                               i11 * 16U),
                    64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 128U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + threadIdx.x / 32U % 2U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(24576U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 128U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(
                    bFrags[i11], sarB + (64U * __anf02 * 16U + i11 * 16U), 64U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 64U);
        uint32_t mcol1 = blockIdx.x % (cols / 64U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 128U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 64U + __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 32U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 4U) * 64U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 64U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 4U) * 128U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 128U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U / 2U) * 128U +
                               __anf01 * 16U + 16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 128U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    float *sarAcc = (float *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 16U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 16U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (16U * (threadIdx.x / 32U) * 128U + __anf01 * 16U +
                               16U * i0 * 16U),
                    16U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 64U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 128U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 4096U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 4096U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 4U) * 32U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 32U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 4U) * 64U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 64U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 4U) * 128U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 128U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U / 2U) * 128U +
                               __anf01 * 16U + 32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 128U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    float *sarAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 32U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 32U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (32U * (threadIdx.x / 32U) * 128U + __anf01 * 16U +
                               32U * i0 * 16U),
                    32U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 64U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 128U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 4U);
    uint32_t fi = 0U;
    for (; fi < 4U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 4096U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 4096U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 4U) * 32U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 4U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 32U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 32U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 32U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 32U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 32U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 4U) * 64U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 64U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 64U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 64U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 4U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 64U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 4U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 64U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        2U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 4U) * 128U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 2U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 4U * 32U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 2U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 2U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 16U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 4U * 128U +
                                 __anf01 / 2U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 4U * 32U +
                                 __anf01 % 2U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        4U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U / 2U) * 128U +
                               __anf01 * 16U + 64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 4U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U +
                               threadIdx.x / 32U % 2U * 64U + i11 * 16U),
                    128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 32U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U / 2U * 128U +
                                 __anf01 / 4U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + threadIdx.x / 32U % 2U * 64U +
                                 __anf01 % 4U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8_0(uint32_t cols,
    uint32_t shared, __nv_bfloat16 *gA, __nv_bfloat16 *gB, uint32_t nthr,
    __nv_bfloat16 *gC, float beta, float alpha, __nv_bfloat16 *gD)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    float *sarAcc = (float *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t num_k_tiles = shared / 64U;
    auto &aFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_a, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &bFrags = KPR_INIT_ARR(kpr_fragment(wmma::matrix_b, 16U, 16U, 16U,
                                    __nv_bfloat16, wmma::row_major),
        8U);
    auto &accFrags = KPR_INIT_ARR(
        kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 64U);
    uint32_t fi = 0U;
    for (; fi < 64U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local, gA + (shared * mrow * 128U + __anf0 * 64U +
                                       shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 128U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                    sarA + (64U * (threadIdx.x / 32U) * 128U + __anf01 * 16U +
                               64U * i0 * 16U),
                    64U);
            uint32_t __anf02 = dotIdx;
            uint32_t i11 = 0U;
            for (; i11 < 8U; i11++)
                wmma::load_matrix_sync(bFrags[i11],
                    sarB + (128U * __anf02 * 16U + i11 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    auto &acc_frag = accFrags[resIdxM * 8U + resIdxN];
                    wmma::mma_sync(
                        acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
                }
            }
        }
    }
    uint32_t idx = 0U;
    for (; idx < 64U; idx++) {
        uint32_t mrow1 = blockIdx.x / (cols / 128U);
        uint32_t mcol1 = blockIdx.x % (cols / 128U);
        wmma::store_matrix_sync(sarAcc + 16U * (threadIdx.x / 32U) * 16U,
            accFrags[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf01 = idx;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < 256U; flat += 32U) {
            uint32_t __anf02 = flat;
            uint32_t row = __anf02 / 16U;
            uint32_t col = __anf02 % 16U;
            uint32_t globalRow = mrow1 * 128U + threadIdx.x / 32U * 128U +
                                 __anf01 / 8U * 16U + row;
            uint32_t globalCol = mcol1 * 128U + __anf01 % 8U * 16U + col;
            float av = sarAcc[(16U * (threadIdx.x / 32U) + row) * 16U + col];
            gD[globalRow * cols + globalCol] = __float2bfloat16(
                beta * __bfloat162float(gC[globalRow * cols + globalCol]) +
                alpha * av);
        }
    }
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2_0, nblk,
        128U, 8192U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4_0, nblk, 64U,
        6144U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2_0, nblk, 64U,
        6144U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(5120U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4_0, nblk, 32U,
        5120U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2_0, nblk,
        128U, 12288U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4_0, nblk, 64U,
        10240U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2_0, nblk, 64U,
        10240U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(9216U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4_0, nblk, 32U,
        9216U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2_0, nblk,
        128U, 20480U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4_0, nblk, 64U,
        18432U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2_0, nblk, 64U,
        18432U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(17408U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4_0, nblk, 32U,
        17408U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4_0, nblk,
        128U, 10240U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8_0, nblk,
        64U, 8192U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2_0, nblk,
        128U, 10240U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4_0, nblk,
        64U, 8192U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(7168U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8_0, nblk,
        32U, 7168U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2_0, nblk,
        256U, 20480U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4_0, nblk,
        128U, 16384U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8_0, nblk,
        64U, 14336U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2_0, nblk,
        128U, 16384U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4_0, nblk,
        64U, 14336U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(13312U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8_0, nblk,
        32U, 13312U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2_0, nblk,
        256U, 32768U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4_0, nblk,
        128U, 28672U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8_0, nblk,
        64U, 26624U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2_0, nblk,
        128U, 28672U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4_0, nblk,
        64U, 26624U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 64U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(25600U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8_0, nblk,
        32U, 25600U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4_0, nblk,
        128U, 10240U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2_0, nblk,
        128U, 10240U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4_0, nblk,
        64U, 8192U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2_0, nblk,
        64U, 8192U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(7168U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4_0, nblk,
        32U, 7168U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2_0, nblk,
        256U, 20480U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4_0, nblk,
        128U, 16384U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2_0, nblk,
        128U, 16384U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4_0, nblk,
        64U, 14336U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(14336U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2_0, nblk,
        64U, 14336U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(13312U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4_0, nblk,
        32U, 13312U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2_0, nblk,
        256U, 32768U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4_0, nblk,
        128U, 28672U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(28672U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2_0, nblk,
        128U, 28672U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4_0, nblk,
        64U, 26624U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(26624U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2_0, nblk,
        64U, 26624U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 128U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(25600U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4_0, nblk,
        32U, 25600U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4_0, nblk,
        256U, 16384U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8_0, nblk,
        128U, 12288U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2_0, nblk,
        256U, 16384U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4_0, nblk,
        128U, 12288U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8_0, nblk,
        64U, 10240U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2_0, nblk,
        128U, 12288U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4_0, nblk,
        64U, 10240U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(9216U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8_0, nblk,
        32U, 9216U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2_0, nblk,
        512U, 32768U, s, cols, shared, gA, gB, 512U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0, nblk,
        256U, 24576U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8_0, nblk,
        128U, 20480U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2_0, nblk,
        256U, 24576U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4_0, nblk,
        128U, 20480U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8_0, nblk,
        64U, 18432U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2_0, nblk,
        128U, 20480U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(18432U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4_0, nblk,
        64U, 18432U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(17408U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8_0, nblk,
        32U, 17408U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(
        __hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2_0,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2_0, nblk,
        512U, 49152U, s, cols, shared, gA, gB, 512U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4_0, nblk,
        256U, 40960U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(36864U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8_0, nblk,
        128U, 36864U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2_0, nblk,
        256U, 40960U, s, cols, shared, gA, gB, 256U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(36864U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4_0, nblk,
        128U, 36864U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(34816U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8_0, nblk,
        64U, 34816U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(36864U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2_0, nblk,
        128U, 36864U, s, cols, shared, gA, gB, 128U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(34816U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4_0, nblk,
        64U, 34816U, s, cols, shared, gA, gB, 64U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(true);
    KPR_ASSERT(true);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(33792U);
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8_0, nblk,
        32U, 33792U, s, cols, shared, gA, gB, 32U, gC, beta, alpha, gD);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
