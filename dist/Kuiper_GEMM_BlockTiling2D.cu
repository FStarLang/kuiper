
#include "Kuiper_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x8_8x8_rr
*/
static void
__hoisted_0(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(2048U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 8U; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = threadIdx.x * 4U;
        for (; i1 < 512U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = i1 / 8U;
            uint32_t col = i1 % 8U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 8U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k += 1U)
                sA[row * 8U + col + k] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = threadIdx.x * 4U;
        for (; i < 512U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = i / 64U;
            uint32_t col = i % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 8U) + mcol * 64U + cols * row +
                       col);
            uint32_t k = 0U;
            for (; k < 4U; k += 1U)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 8U; dotIdx += 1U) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U)
                rAcol[i0] = sA[(threadIdx.x / 8U * 8U + i0) * 8U + dotIdx];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM += 1U) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN += 1U)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rr(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 8U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_KCALL(__hoisted_0,
              rows / 64U * (cols / 64U),
              64U, 4096U, alpha, beta, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x8_8x8_rr
*/
static void
__hoisted_1(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 8U; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = threadIdx.x * 4U;
        for (; i1 < 1024U; i1 += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = i1 / 8U;
            uint32_t col = i1 % 8U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 8U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k += 1U)
                sA[row * 8U + col + k] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = threadIdx.x * 4U;
        for (; i < 1024U; i += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 8U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 8U; dotIdx += 1U) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U)
                rAcol[i0] = sA[(threadIdx.x / 16U * 8U + i0) * 8U + dotIdx];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM += 1U) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN += 1U)
            t_tile[(blockIdx.x / (cols / 128U) * 128U + threadIdx.x / 16U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 16U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t *gA,
                                                      float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 8U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_1,
              rows / 128U * (cols / 128U),
              256U, 8192U, alpha, beta, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x8_8x8_cr
*/
static void
__hoisted_2(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 8U; bkIdx += 1U) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = threadIdx.x * 4U;
        for (; i1 < 1024U; i1 += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = i1 / 8U;
            uint32_t col = i1 % 8U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 8U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k += 1U)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = threadIdx.x * 4U;
        for (; i < 1024U; i += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = i / 128U;
            uint32_t col = i % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 8U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k += 1U)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 8U; dotIdx += 1U) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0 += 1U)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1 += 1U)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM += 1U) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN += 1U)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM += 1U) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN += 1U)
            t_tile[(blockIdx.x / (cols / 128U) * 128U + threadIdx.x / 16U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 16U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t *gA,
                                                      float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 8U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_KCALL(__hoisted_2,
              rows / 128U * (cols / 128U),
              256U, 8192U, alpha, beta, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}
