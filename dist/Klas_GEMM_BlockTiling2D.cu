
#include "Klas_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x32x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_8x8_0,
              rows / 32U * (cols / 32U),
              16U, 8192U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x32x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_8x16_0,
              rows / 32U * (cols / 32U),
              8U, 8192U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x32x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_16x8_0,
              rows / 32U * (cols / 32U),
              8U, 8192U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x32x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_16x16_0,
              rows / 32U * (cols / 32U),
              4U, 8192U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x32x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_8x8_0,
              rows / 32U * (cols / 32U),
              16U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x32x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_8x16_0,
              rows / 32U * (cols / 32U),
              8U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x32x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_16x8_0,
              rows / 32U * (cols / 32U),
              8U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x32x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_16x16_0,
              rows / 32U * (cols / 32U),
              4U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x64x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_8x8_0,
              rows / 32U * (cols / 64U),
              32U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x64x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_8x16_0,
              rows / 32U * (cols / 64U),
              16U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x64x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_16x8_0,
              rows / 32U * (cols / 64U),
              16U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x64x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_16x16_0,
              rows / 32U * (cols / 64U),
              8U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x64x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_8x8_0,
              rows / 32U * (cols / 64U),
              32U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x64x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_8x16_0,
              rows / 32U * (cols / 64U),
              16U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x64x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_16x8_0,
              rows / 32U * (cols / 64U),
              16U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x64x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_16x16_0,
              rows / 32U * (cols / 64U),
              8U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x128x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_8x8_0,
              rows / 32U * (cols / 128U),
              64U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x128x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_8x16_0,
              rows / 32U * (cols / 128U),
              32U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x128x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_16x8_0,
              rows / 32U * (cols / 128U),
              32U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x128x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_16x16_0,
              rows / 32U * (cols / 128U),
              16U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x128x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_8x8_0,
              rows / 32U * (cols / 128U),
              64U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x128x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_8x16_0,
              rows / 32U * (cols / 128U),
              32U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x128x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_16x8_0,
              rows / 32U * (cols / 128U),
              32U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x128x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 32U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 32U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(32U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_16x16_0,
              rows / 32U * (cols / 128U),
              16U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x32x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_8x8_0,
              rows / 64U * (cols / 32U),
              32U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x32x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_8x16_0,
              rows / 64U * (cols / 32U),
              16U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x32x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_16x8_0,
              rows / 64U * (cols / 32U),
              16U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x32x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_16x16_0,
              rows / 64U * (cols / 32U),
              8U, 12288U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x32x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_8x8_0,
              rows / 64U * (cols / 32U),
              32U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x32x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_8x16_0,
              rows / 64U * (cols / 32U),
              16U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x32x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_16x8_0,
              rows / 64U * (cols / 32U),
              16U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x32x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_16x16_0,
              rows / 64U * (cols / 32U),
              8U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x64x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_8x8_0,
              rows / 64U * (cols / 64U),
              64U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x64x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_8x16_0,
              rows / 64U * (cols / 64U),
              32U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x64x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_16x8_0,
              rows / 64U * (cols / 64U),
              32U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x64x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_16x16_0,
              rows / 64U * (cols / 64U),
              16U, 16384U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x64x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_8x8_0,
              rows / 64U * (cols / 64U),
              64U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x64x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_8x16_0,
              rows / 64U * (cols / 64U),
              32U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x64x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_16x8_0,
              rows / 64U * (cols / 64U),
              32U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x64x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_16x16_0,
              rows / 64U * (cols / 64U),
              16U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x128x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_8x8_0,
              rows / 64U * (cols / 128U),
              128U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x128x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_8x16_0,
              rows / 64U * (cols / 128U),
              64U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x128x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_16x8_0,
              rows / 64U * (cols / 128U),
              64U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x128x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_16x16_0,
              rows / 64U * (cols / 128U),
              32U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x128x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_8x8_0,
              rows / 64U * (cols / 128U),
              128U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x128x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_8x16_0,
              rows / 64U * (cols / 128U),
              64U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x128x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_16x8_0,
              rows / 64U * (cols / 128U),
              64U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x128x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 64U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 64U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(64U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_16x16_0,
              rows / 64U * (cols / 128U),
              32U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x32x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_8x8_0,
              rows / 128U * (cols / 32U),
              64U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x32x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_8x16_0,
              rows / 128U * (cols / 32U),
              32U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x32x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_16x8_0,
              rows / 128U * (cols / 32U),
              32U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x32x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_16x16_0,
              rows / 128U * (cols / 32U),
              16U, 20480U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x32x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_8x8_0,
              rows / 128U * (cols / 32U),
              64U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x32x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    8U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        8U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_8x16_0,
              rows / 128U * (cols / 32U),
              32U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x32x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) + 8U * (threadIdx.x % 4U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       8U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_16x8_0,
              rows / 128U * (cols / 32U),
              32U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x32x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 32U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                    16U * (threadIdx.x / 2U) + resIdxM) * cols +
                   32U * (blockIdx.x % (cols / 32U)) +
                   16U * (threadIdx.x % 2U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 32U)) +
                        16U * (threadIdx.x / 2U) + resIdxM) * cols +
                       32U * (blockIdx.x % (cols / 32U)) +
                       16U * (threadIdx.x % 2U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_16x16_0,
              rows / 128U * (cols / 32U),
              16U, 40960U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x64x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_8x8_0,
              rows / 128U * (cols / 64U),
              128U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x64x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_8x16_0,
              rows / 128U * (cols / 64U),
              64U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x64x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_16x8_0,
              rows / 128U * (cols / 64U),
              64U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x64x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_16x16_0,
              rows / 128U * (cols / 64U),
              32U, 24576U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x64x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t shared,
                                     uint32_t cols,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_8x8_0,
              rows / 128U * (cols / 64U),
              128U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x64x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    8U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        8U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_8x16_0,
              rows / 128U * (cols / 64U),
              64U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x64x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) + 8U * (threadIdx.x % 8U) +
                   resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       8U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_16x8_0,
              rows / 128U * (cols / 64U),
              64U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x64x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 64U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                    16U * (threadIdx.x / 4U) + resIdxM) * cols +
                   64U * (blockIdx.x % (cols / 64U)) +
                   16U * (threadIdx.x % 4U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 64U)) +
                        16U * (threadIdx.x / 4U) + resIdxM) * cols +
                       64U * (blockIdx.x % (cols / 64U)) +
                       16U * (threadIdx.x % 4U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_16x16_0,
              rows / 128U * (cols / 64U),
              32U, 49152U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x128x32_8x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_8x8_0,
              rows / 128U * (cols / 128U),
              256U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x128x32_8x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_8x16_0,
              rows / 128U * (cols / 128U),
              128U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x128x32_16x8_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_16x8_0,
              rows / 128U * (cols / 128U),
              128U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x128x32_16x16_0(float alpha,
                                        float beta,
                                        uint32_t shared,
                                        uint32_t cols,
                                        float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 32U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 32U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 32U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 32U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_16x16_0,
              rows / 128U * (cols / 128U),
              64U, 32768U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x128x64_8x8_0(float alpha,
                                      float beta,
                                      uint32_t shared,
                                      uint32_t cols,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_8x8_0,
              rows / 128U * (cols / 128U),
              256U, 65536U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x128x64_8x16_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[8U];
            memset(rAcol, 0U, 8U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    8U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        8U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_8x16_0,
              rows / 128U * (cols / 128U),
              128U, 65536U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x128x64_16x8_0(float alpha,
                                       float beta,
                                       uint32_t shared,
                                       uint32_t cols,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[8U];
            memset(rBrow, 0U, 8U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 16U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   8U * (threadIdx.x % 16U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 16U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       8U * (threadIdx.x % 16U) + resIdxN]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_16x8_0,
              rows / 128U * (cols / 128U),
              128U, 65536U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x128x64_16x16_0(float alpha,
                                        float beta,
                                        uint32_t shared,
                                        uint32_t cols,
                                        float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 64U; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        float *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i0 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 4U) % 64U;
            vec_memcpy(local,
                       tileA + shared * (mrow * 128U) + __anf03 * 64U +
                       shared * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sA[(col + k) * 128U + row] = local[k];
        }
        uint32_t __anf06 = bkIdx;
        float *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 128U;
            uint32_t col = (i + threadIdx.x * 4U) % 128U;
            vec_memcpy(local,
                       tileB + cols * (__anf06 * 64U) + mcol * 128U +
                       cols * row + col);
            uint32_t k = 0U;
            for (; k < 4U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            float rAcol[16U];
            memset(rAcol, 0U, 16U * sizeof(float));
            float rBrow[16U];
            memset(rBrow, 0U, 16U * sizeof(float));
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    rchProd[idx] += rAcol[resIdxM] * rBrow[resIdxN];
                }
            }
        }
    }
    float *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++)
            t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                    16U * (threadIdx.x / 8U) + resIdxM) * cols +
                   128U * (blockIdx.x % (cols / 128U)) +
                   16U * (threadIdx.x % 8U) + resIdxN]
                = beta *
                t_tile[(128U * (blockIdx.x / (cols / 128U)) +
                        16U * (threadIdx.x / 8U) + resIdxM) * cols +
                       128U * (blockIdx.x % (cols / 128U)) +
                       16U * (threadIdx.x % 8U) + resIdxN]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_16x16_0,
              rows / 128U * (cols / 128U),
              64U, 65536U, alpha, beta, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
