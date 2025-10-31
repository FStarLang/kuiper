
#include "Kuiper_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x8
*/
static void
__hoisted_0(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_0, rows / 32U * (cols / 32U), 16U, 8192U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x16
*/
static void
__hoisted_1(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 2U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 2U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 2U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_1, rows / 32U * (cols / 32U), 8U, 8192U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x8
*/
static void
__hoisted_2(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_2, rows / 32U * (cols / 32U), 8U, 8192U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x16
*/
static void
__hoisted_3(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 16U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 16U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 2U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 2U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 2U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_3, rows / 32U * (cols / 32U), 4U, 8192U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x8
*/
static void
__hoisted_4(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_4, rows / 32U * (cols / 32U), 16U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x16
*/
static void
__hoisted_5(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 2U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 2U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 2U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_5, rows / 32U * (cols / 32U), 8U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x8
*/
static void
__hoisted_6(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_6, rows / 32U * (cols / 32U), 8U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x16
*/
static void
__hoisted_7(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 16U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 16U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 2U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 32U + threadIdx.x / 2U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 32U +
                        threadIdx.x / 2U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_7, rows / 32U * (cols / 32U), 4U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x8
*/
static void
__hoisted_8(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
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
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_8, rows / 32U * (cols / 64U), 32U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x16
*/
static void
__hoisted_9(float_t alpha,
            float_t beta,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_9, rows / 32U * (cols / 64U), 16U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x8
*/
static void
__hoisted_10(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_10, rows / 32U * (cols / 64U), 16U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x16
*/
static void
__hoisted_11(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_11, rows / 32U * (cols / 64U), 8U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x8
*/
static void
__hoisted_12(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
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
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_12, rows / 32U * (cols / 64U), 32U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x16
*/
static void
__hoisted_13(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_13, rows / 32U * (cols / 64U), 16U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x8
*/
static void
__hoisted_14(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_14, rows / 32U * (cols / 64U), 16U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x16
*/
static void
__hoisted_15(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 32U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 32U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_15, rows / 32U * (cols / 64U), 8U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x8
*/
static void
__hoisted_16(float_t alpha,
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
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 16U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 16U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_16, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_16, rows / 32U * (cols / 128U), 64U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x16
*/
static void
__hoisted_17(float_t alpha,
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
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_17, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_17, rows / 32U * (cols / 128U), 32U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x8
*/
static void
__hoisted_18(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 16U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 16U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 16U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_18, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_18, rows / 32U * (cols / 128U), 32U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x16
*/
static void
__hoisted_19(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_19, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_19, rows / 32U * (cols / 128U), 16U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x8
*/
static void
__hoisted_20(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
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
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 16U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 16U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_20, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_20, rows / 32U * (cols / 128U), 64U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x16
*/
static void
__hoisted_21(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_21, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_21, rows / 32U * (cols / 128U), 32U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x8
*/
static void
__hoisted_22(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 16U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 16U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 16U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_22, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_22, rows / 32U * (cols / 128U), 32U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x16
*/
static void
__hoisted_23(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 32U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 32U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 32U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_23, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_23, rows / 32U * (cols / 128U), 16U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x8
*/
static void
__hoisted_24(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_24, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_24, rows / 64U * (cols / 32U), 32U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x16
*/
static void
__hoisted_25(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 2U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 2U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 2U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_25, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_25, rows / 64U * (cols / 32U), 16U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x8
*/
static void
__hoisted_26(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_26, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_26, rows / 64U * (cols / 32U), 16U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x16
*/
static void
__hoisted_27(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 2U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 2U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 2U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute
         (__hoisted_27, cudaFuncAttributeMaxDynamicSharedMemorySize, 12288U));
    KPR_KCALL(__hoisted_27, rows / 64U * (cols / 32U), 8U, 12288U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x8
*/
static void
__hoisted_28(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_28, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_28, rows / 64U * (cols / 32U), 32U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x16
*/
static void
__hoisted_29(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 2U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 2U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 2U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_29, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_29, rows / 64U * (cols / 32U), 16U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x8
*/
static void
__hoisted_30(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_30, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_30, rows / 64U * (cols / 32U), 16U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x16
*/
static void
__hoisted_31(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 2U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 64U + threadIdx.x / 2U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 64U +
                        threadIdx.x / 2U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_31, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_31, rows / 64U * (cols / 32U), 8U, 24576U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x8
*/
static void
__hoisted_32(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
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
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_32, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_32, rows / 64U * (cols / 64U), 64U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x16
*/
static void
__hoisted_33(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_33, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_33, rows / 64U * (cols / 64U), 32U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x8
*/
static void
__hoisted_34(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_34, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_34, rows / 64U * (cols / 64U), 32U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x16
*/
static void
__hoisted_35(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_35, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_35, rows / 64U * (cols / 64U), 16U, 16384U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x8
*/
static void
__hoisted_36(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
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
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8(float_t alpha,
                                                  float_t beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float_t *gA,
                                                  float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_36, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_36, rows / 64U * (cols / 64U), 64U, 32768U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x16
*/
static void
__hoisted_37(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x16(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_37, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_37, rows / 64U * (cols / 64U), 32U, 32768U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x8
*/
static void
__hoisted_38(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_38, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_38, rows / 64U * (cols / 64U), 32U, 32768U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x16
*/
static void
__hoisted_39(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 64U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 64U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_39, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_39, rows / 64U * (cols / 64U), 16U, 32768U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x8
*/
static void
__hoisted_40(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
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
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 16U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 16U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_40, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_40, rows / 64U * (cols / 128U), 128U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x16
*/
static void
__hoisted_41(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_41, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_41, rows / 64U * (cols / 128U), 64U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x8
*/
static void
__hoisted_42(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 16U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 16U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 16U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_42, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_42, rows / 64U * (cols / 128U), 64U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x16
*/
static void
__hoisted_43(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_43, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_43, rows / 64U * (cols / 128U), 32U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x8
*/
static void
__hoisted_44(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
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
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 16U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 16U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_44, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_44, rows / 64U * (cols / 128U), 128U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x16
*/
static void
__hoisted_45(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_45, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_45, rows / 64U * (cols / 128U), 64U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x8
*/
static void
__hoisted_46(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 16U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 16U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 16U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 16U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_46, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_46, rows / 64U * (cols / 128U), 64U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x16
*/
static void
__hoisted_47(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 64U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 64U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 64U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_47, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_47, rows / 64U * (cols / 128U), 32U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x8
*/
static void
__hoisted_48(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_48, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_48, rows / 128U * (cols / 32U), 64U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x16
*/
static void
__hoisted_49(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 2U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 2U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 2U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_49, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_49, rows / 128U * (cols / 32U), 32U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x8
*/
static void
__hoisted_50(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_50, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_50, rows / 128U * (cols / 32U), 32U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x16
*/
static void
__hoisted_51(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 2U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 2U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 2U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute
         (__hoisted_51, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480U));
    KPR_KCALL(__hoisted_51, rows / 128U * (cols / 32U), 16U, 20480U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x8
*/
static void
__hoisted_52(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_52, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_52, rows / 128U * (cols / 32U), 64U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x16
*/
static void
__hoisted_53(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 2U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 2U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 2U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_53, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_53, rows / 128U * (cols / 32U), 32U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x8
*/
static void
__hoisted_54(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 4U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 4U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U + threadIdx.x % 4U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_54, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_54, rows / 128U * (cols / 32U), 32U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x16
*/
static void
__hoisted_55(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 2U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 32U + threadIdx.x % 2U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 32U) * 128U + threadIdx.x / 2U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 32U) * 32U +
                   threadIdx.x % 2U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 32U) * 128U +
                        threadIdx.x / 2U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 32U) * 32U +
                       threadIdx.x % 2U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute
         (__hoisted_55, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960U));
    KPR_KCALL(__hoisted_55, rows / 128U * (cols / 32U), 16U, 40960U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x8
*/
static void
__hoisted_56(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
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
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_56, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_56, rows / 128U * (cols / 64U), 128U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x16
*/
static void
__hoisted_57(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_57, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_57, rows / 128U * (cols / 64U), 64U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x8
*/
static void
__hoisted_58(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_58, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_58, rows / 128U * (cols / 64U), 64U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x16
*/
static void
__hoisted_59(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_59, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_59, rows / 128U * (cols / 64U), 32U, 24576U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x8
*/
static void
__hoisted_60(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
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
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8(float_t alpha,
                                                   float_t beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t *gA,
                                                   float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_60, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_60, rows / 128U * (cols / 64U), 128U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x16
*/
static void
__hoisted_61(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 4U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 4U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x16(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_61, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_61, rows / 128U * (cols / 64U), 64U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x8
*/
static void
__hoisted_62(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 8U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 8U * 8U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U + threadIdx.x % 8U * 8U +
                       resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_62, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_62, rows / 128U * (cols / 64U), 64U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x16
*/
static void
__hoisted_63(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 4U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 64U + threadIdx.x % 4U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 64U) * 128U + threadIdx.x / 4U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 64U) * 64U +
                   threadIdx.x % 4U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 64U) * 128U +
                        threadIdx.x / 4U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 64U) * 64U +
                       threadIdx.x % 4U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute
         (__hoisted_63, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152U));
    KPR_KCALL(__hoisted_63, rows / 128U * (cols / 64U), 32U, 49152U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x8
*/
static void
__hoisted_64(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
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
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_64, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_64, rows / 128U * (cols / 128U), 256U, 32768U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x16
*/
static void
__hoisted_65(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 128U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_65, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_65, rows / 128U * (cols / 128U), 128U, 32768U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x8
*/
static void
__hoisted_66(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 16U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 128U +
                    threadIdx.x / 16U * 16U + resIdxM) * cols +
                   blockIdx.x % (cols / 128U) * 128U + threadIdx.x % 16U * 8U +
                   resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 16U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_66, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_66, rows / 128U * (cols / 128U), 128U, 32768U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x16
*/
static void
__hoisted_67(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 128U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x16(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t *gA,
                                                      float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute
         (__hoisted_67, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768U));
    KPR_KCALL(__hoisted_67, rows / 128U * (cols / 128U), 64U, 32768U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x8
*/
static void
__hoisted_68(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
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
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 16U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x8(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t *gA,
                                                    float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute
         (__hoisted_68, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536U));
    KPR_KCALL(__hoisted_68, rows / 128U * (cols / 128U), 256U, 65536U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x16
*/
static void
__hoisted_69(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[8U];
    memset(rAcol, 0U, 8U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 8U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 128U + threadIdx.x / 8U * 8U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 8U * 8U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x16(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute
         (__hoisted_69, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536U));
    KPR_KCALL(__hoisted_69, rows / 128U * (cols / 128U), 128U, 65536U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x8
*/
static void
__hoisted_70(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[8U];
    memset(rBrow, 0U, 8U * sizeof(float_t));
    float_t rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 16U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 8U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 16U * 8U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++)
                    rchProd[resIdxM * 8U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 128U +
                    threadIdx.x / 16U * 16U + resIdxM) * cols +
                   blockIdx.x % (cols / 128U) * 128U + threadIdx.x % 16U * 8U +
                   resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 16U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 16U * 8U + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x8(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t *gA,
                                                     float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute
         (__hoisted_70, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536U));
    KPR_KCALL(__hoisted_70, rows / 128U * (cols / 128U), 128U, 65536U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x16
*/
static void
__hoisted_71(float_t alpha,
             float_t beta,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    float_t *sA = (float_t *) KPR_SHMEM_AT(0U);
    float_t *sB = (float_t *) KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float_t rAcol[16U];
    memset(rAcol, 0U, 16U * sizeof(float_t));
    float_t rBrow[16U];
    memset(rBrow, 0U, 16U * sizeof(float_t));
    float_t rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float_t));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        float_t *tileA = gA;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf01 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        float_t *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float_t local[4U];
            memset(local, 0U, 4U * sizeof(float_t));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf01 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            uint32_t i0 = 0U;
            for (; i0 < 16U; i0++)
                rAcol[i0] = sA[dotIdx * 128U + threadIdx.x / 8U * 16U + i0];
            uint32_t i1 = 0U;
            for (; i1 < 16U; i1++)
                rBrow[i1] = sB[dotIdx * 128U + threadIdx.x % 8U * 16U + i1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++)
                    rchProd[resIdxM * 16U + resIdxN] +=
                        rAcol[resIdxM] * rBrow[resIdxN];
            }
        }
    }
    float_t *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(blockIdx.x / (cols / 128U) * 128U + threadIdx.x / 8U * 16U +
                    resIdxM) * cols + blockIdx.x % (cols / 128U) * 128U +
                   threadIdx.x % 8U * 16U + resIdxN]
                = beta *
                t_tile[(blockIdx.x / (cols / 128U) * 128U +
                        threadIdx.x / 8U * 16U + resIdxM) * cols +
                       blockIdx.x % (cols / 128U) * 128U +
                       threadIdx.x % 8U * 16U + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x16(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t *gA,
                                                      float_t *gB, float_t *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute
         (__hoisted_71, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536U));
    KPR_KCALL(__hoisted_71, rows / 128U * (cols / 128U), 64U, 65536U, alpha,
              beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
