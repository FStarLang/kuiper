
#include "Klas_UGS_SwiGLU.h"

__device__ static half epilogue_cell(half gate_h, half up_h)
{
    float g = __half2float(gate_h);
    return __hmul(__float2half_rn(g * (1.0f / (1.0f + expf(0.0f - g)))), up_h);
}

__global__
/**
  hoisted when extracting swiglu
*/
static void __hoisted_swiglu_0(uint32_t k, uint32_t n, half *gA, half *gW,
                               float *gPf32)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 16U;
    uint32_t num_n_tiles = 2U * n / 64U;
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
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float),
                     16U);
    uint32_t fi = 0U;
    for (; fi < 16U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
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
                       tileA + (k * mrow * 64U + __anf03 * 16U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[row * 16U + col + k1] = local[k1];
        }
        half *tileB = gW;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (2U * n * __anf03 * 16U + mcol * 64U +
                                2U * n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
                                       (16U * (threadIdx.x / 32U) * 64U +
                                        __anf010 * 16U + 16U * i0 * 16U), 16U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (64U * __anf011 * 16U + i1 * 16U), 64U);
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
            wmma::store_matrix_sync(gPf32 +
                                    (2U * n * (blockIdx.x / (2U * n / 64U)) *
                                     64U + blockIdx.x % (2U * n / 64U) * 64U +
                                     2U * n * (threadIdx.x / 32U) * 64U +
                                     2U * n * i * 16U + j * 16U),
                                    accFrags[i * 4U + j], 2U * n,
                                    wmma::mem_row_major);
            __syncwarp();
        }
    }
}

__global__
/**
  hoisted when extracting swiglu
*/
static void __hoisted_swiglu_1(uint32_t m, uint32_t n, float *gPf32, half *gP)
{
    uint32_t k1 = 0U;
    uint32_t kend = m * 2U * n;
    for (; k1 < kend; KPR_ASSERT(k1 <= kend)) {
        uint32_t kv = k1;
        uint32_t ci = kv % (2U * n);
        uint32_t ri1 = kv / (2U * n);
        gP[ri1 * 2U * n + ci] = __float2half_rn(gPf32[ri1 * 2U * n + ci]);
        k1 = kv + 1U;
    }
}

__global__
/**
  hoisted when extracting swiglu
*/
static void __hoisted_swiglu_2(uint32_t m, uint32_t n, half *gP, half *gOut)
{
    uint32_t k1 = 0U;
    uint32_t kend = m * n;
    for (; k1 < kend; KPR_ASSERT(k1 <= kend)) {
        uint32_t kv = k1;
        uint32_t ci = kv % n;
        uint32_t ri1 = kv / n;
        uint32_t gc0 = 128U * (ci / 64U) + ci % 64U;
        gOut[ri1 * n + ci] =
            epilogue_cell(gP[ri1 * 2U * n + gc0], gP[ri1 * 2U * n + gc0 + 64U]);
        k1 = kv + 1U;
    }
}

void
Klas_UGS_SwiGLU_swiglu(uint32_t m,
                       uint32_t k,
                       uint32_t n,
                       half *gA, half *gW, float *gPf32, half *gP, half *gOut)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 16U == 0U);
    KPR_GUARD(2U * n % 64U == 0U);
    uint32_t nblk = m / 64U * (2U * n / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_swiglu_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_swiglu_0, nblk, 32U, 4096U, s, k, n, gA, gW, gPf32);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_swiglu_1, 1U, 1U, 0U, s0, m, n, gPf32, gP);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_swiglu_2, 1U, 1U, 0U, s1, m, n, gP, gOut);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
