
#include "Klas_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x32x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_8x8_0,
              m / 32U * (n / 32U),
              16U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x32_8x8
*/
static void
__hoisted_g_gemm_bf16_32x32x32_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_8x8_0,
              m / 32U * (n / 32U),
              16U, 4096U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x32x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_8x16_0,
              m / 32U * (n / 32U), 8U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x32_8x16
*/
static void
__hoisted_g_gemm_bf16_32x32x32_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_8x16_0,
              m / 32U * (n / 32U), 8U, 4096U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x32x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_16x8_0,
              m / 32U * (n / 32U), 8U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x32_16x8
*/
static void
__hoisted_g_gemm_bf16_32x32x32_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_16x8_0,
              m / 32U * (n / 32U), 8U, 4096U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x32x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_16x16_0,
              m / 32U * (n / 32U), 4U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x32_16x16
*/
static void
__hoisted_g_gemm_bf16_32x32x32_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 32U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_16x16_0,
              m / 32U * (n / 32U), 4U, 4096U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x32x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_8x8_0,
              m / 32U * (n / 32U),
              16U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x64_8x8
*/
static void
__hoisted_g_gemm_bf16_32x32x64_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_8x8_0,
              m / 32U * (n / 32U),
              16U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x32x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_8x16_0,
              m / 32U * (n / 32U),
              8U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x64_8x16
*/
static void
__hoisted_g_gemm_bf16_32x32x64_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_8x16_0,
              m / 32U * (n / 32U), 8U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x32x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_16x8_0,
              m / 32U * (n / 32U),
              8U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x64_16x8
*/
static void
__hoisted_g_gemm_bf16_32x32x64_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_16x8_0,
              m / 32U * (n / 32U), 8U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x32x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_16x16_0,
              m / 32U * (n / 32U),
              4U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x32x64_16x16
*/
static void
__hoisted_g_gemm_bf16_32x32x64_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 32U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_16x16_0,
              m / 32U * (n / 32U), 4U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x64x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_8x8_0,
              m / 32U * (n / 64U),
              32U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x32_8x8
*/
static void
__hoisted_g_gemm_bf16_32x64x32_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_8x8_0,
              m / 32U * (n / 64U),
              32U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x64x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_8x16_0,
              m / 32U * (n / 64U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x32_8x16
*/
static void
__hoisted_g_gemm_bf16_32x64x32_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_8x16_0,
              m / 32U * (n / 64U),
              16U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x64x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_16x8_0,
              m / 32U * (n / 64U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x32_16x8
*/
static void
__hoisted_g_gemm_bf16_32x64x32_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_16x8_0,
              m / 32U * (n / 64U),
              16U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x64x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_16x16_0,
              m / 32U * (n / 64U),
              8U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x32_16x16
*/
static void
__hoisted_g_gemm_bf16_32x64x32_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_16x16_0,
              m / 32U * (n / 64U), 8U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x64x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_8x8_0,
              m / 32U * (n / 64U),
              32U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x64_8x8
*/
static void
__hoisted_g_gemm_bf16_32x64x64_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_8x8_0,
              m / 32U * (n / 64U),
              32U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x64x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_8x16_0,
              m / 32U * (n / 64U),
              16U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x64_8x16
*/
static void
__hoisted_g_gemm_bf16_32x64x64_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_8x16_0,
              m / 32U * (n / 64U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x64x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_16x8_0,
              m / 32U * (n / 64U),
              16U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x64_16x8
*/
static void
__hoisted_g_gemm_bf16_32x64x64_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_16x8_0,
              m / 32U * (n / 64U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x64x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_16x16_0,
              m / 32U * (n / 64U),
              8U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x64x64_16x16
*/
static void
__hoisted_g_gemm_bf16_32x64x64_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_16x16_0,
              m / 32U * (n / 64U),
              8U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x128x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_8x8_0,
              m / 32U * (n / 128U),
              64U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x32_8x8
*/
static void
__hoisted_g_gemm_bf16_32x128x32_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_8x8_0,
              m / 32U * (n / 128U),
              64U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x128x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_8x16_0,
              m / 32U * (n / 128U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x32_8x16
*/
static void
__hoisted_g_gemm_bf16_32x128x32_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_8x16_0,
              m / 32U * (n / 128U),
              32U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x128x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_16x8_0,
              m / 32U * (n / 128U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x32_16x8
*/
static void
__hoisted_g_gemm_bf16_32x128x32_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_16x8_0,
              m / 32U * (n / 128U),
              32U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x128x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_16x16_0,
              m / 32U * (n / 128U),
              16U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x32_16x16
*/
static void
__hoisted_g_gemm_bf16_32x128x32_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 1024U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_16x16_0,
              m / 32U * (n / 128U),
              16U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x128x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_8x8_0,
              m / 32U * (n / 128U),
              64U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x64_8x8
*/
static void
__hoisted_g_gemm_bf16_32x128x64_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_8x8_0,
              m / 32U * (n / 128U),
              64U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x128x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_8x16_0,
              m / 32U * (n / 128U),
              32U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x64_8x16
*/
static void
__hoisted_g_gemm_bf16_32x128x64_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_8x16_0,
              m / 32U * (n / 128U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x128x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_16x8_0,
              m / 32U * (n / 128U),
              32U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x64_16x8
*/
static void
__hoisted_g_gemm_bf16_32x128x64_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_16x8_0,
              m / 32U * (n / 128U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x128x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_16x16_0,
              m / 32U * (n / 128U),
              16U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_32x128x64_16x16
*/
static void
__hoisted_g_gemm_bf16_32x128x64_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 32U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(32U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(32U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_16x16_0,
              m / 32U * (n / 128U),
              16U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x32x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_8x8_0,
              m / 64U * (n / 32U),
              32U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x32_8x8
*/
static void
__hoisted_g_gemm_bf16_64x32x32_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_8x8_0,
              m / 64U * (n / 32U),
              32U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x32x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_8x16_0,
              m / 64U * (n / 32U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x32_8x16
*/
static void
__hoisted_g_gemm_bf16_64x32x32_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_8x16_0,
              m / 64U * (n / 32U),
              16U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x32x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_16x8_0,
              m / 64U * (n / 32U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x32_16x8
*/
static void
__hoisted_g_gemm_bf16_64x32x32_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_16x8_0,
              m / 64U * (n / 32U),
              16U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x32x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_16x16_0,
              m / 64U * (n / 32U),
              8U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x32_16x16
*/
static void
__hoisted_g_gemm_bf16_64x32x32_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(6144U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              6144U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_16x16_0,
              m / 64U * (n / 32U), 8U, 6144U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x32x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_8x8_0,
              m / 64U * (n / 32U),
              32U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x64_8x8
*/
static void
__hoisted_g_gemm_bf16_64x32x64_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_8x8_0,
              m / 64U * (n / 32U),
              32U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x32x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_8x16_0,
              m / 64U * (n / 32U),
              16U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x64_8x16
*/
static void
__hoisted_g_gemm_bf16_64x32x64_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_8x16_0,
              m / 64U * (n / 32U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x32x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_16x8_0,
              m / 64U * (n / 32U),
              16U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x64_16x8
*/
static void
__hoisted_g_gemm_bf16_64x32x64_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_16x8_0,
              m / 64U * (n / 32U),
              16U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x32x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_16x16_0,
              m / 64U * (n / 32U),
              8U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x32x64_16x16
*/
static void
__hoisted_g_gemm_bf16_64x32x64_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_16x16_0,
              m / 64U * (n / 32U),
              8U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x64x32_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_8x8_0,
              m / 64U * (n / 64U),
              64U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x32_8x8
*/
static void
__hoisted_g_gemm_bf16_64x64x32_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_8x8_0,
              m / 64U * (n / 64U),
              64U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x64x32_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_8x16_0,
              m / 64U * (n / 64U),
              32U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x32_8x16
*/
static void
__hoisted_g_gemm_bf16_64x64x32_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_8x16_0,
              m / 64U * (n / 64U),
              32U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x64x32_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_16x8_0,
              m / 64U * (n / 64U),
              32U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x32_16x8
*/
static void
__hoisted_g_gemm_bf16_64x64x32_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_16x8_0,
              m / 64U * (n / 64U),
              32U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x64x32_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_16x16_0,
              m / 64U * (n / 64U),
              16U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x32_16x16
*/
static void
__hoisted_g_gemm_bf16_64x64x32_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_16x16_0,
              m / 64U * (n / 64U),
              16U, 8192U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x64x64_8x8_0(float alpha,
                                    float beta,
                                    uint32_t n,
                                    uint32_t k, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_8x8_0,
              m / 64U * (n / 64U),
              64U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x64_8x8
*/
static void
__hoisted_g_gemm_bf16_64x64x64_8x8_0(__nv_bfloat16 alpha,
                                     __nv_bfloat16 beta,
                                     uint32_t n,
                                     uint32_t k,
                                     __nv_bfloat16 *gA,
                                     __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 __nv_bfloat16 *gA,
                                                 __nv_bfloat16 *gB,
                                                 __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_8x8_0,
              m / 64U * (n / 64U),
              64U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x64x64_8x16_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_8x16_0,
              m / 64U * (n / 64U),
              32U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x64_8x16
*/
static void
__hoisted_g_gemm_bf16_64x64x64_8x16_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_8x16_0,
              m / 64U * (n / 64U),
              32U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x64x64_16x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_16x8_0,
              m / 64U * (n / 64U),
              32U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x64_16x8
*/
static void
__hoisted_g_gemm_bf16_64x64x64_16x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_16x8_0,
              m / 64U * (n / 64U),
              32U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x64x64_16x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_16x16_0,
              m / 64U * (n / 64U),
              16U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x64x64_16x16
*/
static void
__hoisted_g_gemm_bf16_64x64x64_16x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_16x16_0,
              m / 64U * (n / 64U),
              16U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x128x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_8x8_0,
              m / 64U * (n / 128U),
              128U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x32_8x8
*/
static void
__hoisted_g_gemm_bf16_64x128x32_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_8x8_0,
              m / 64U * (n / 128U),
              128U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x128x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_8x16_0,
              m / 64U * (n / 128U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x32_8x16
*/
static void
__hoisted_g_gemm_bf16_64x128x32_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_8x16_0,
              m / 64U * (n / 128U),
              64U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x128x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_16x8_0,
              m / 64U * (n / 128U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x32_16x8
*/
static void
__hoisted_g_gemm_bf16_64x128x32_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_16x8_0,
              m / 64U * (n / 128U),
              64U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x128x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_16x16_0,
              m / 64U * (n / 128U),
              32U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x32_16x16
*/
static void
__hoisted_g_gemm_bf16_64x128x32_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 2048U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_16x16_0,
              m / 64U * (n / 128U),
              32U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x128x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_8x8_0,
              m / 64U * (n / 128U),
              128U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x64_8x8
*/
static void
__hoisted_g_gemm_bf16_64x128x64_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 16U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_8x8_0,
              m / 64U * (n / 128U),
              128U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x128x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_8x16_0,
              m / 64U * (n / 128U),
              64U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x64_8x16
*/
static void
__hoisted_g_gemm_bf16_64x128x64_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_8x16_0,
              m / 64U * (n / 128U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x128x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_16x8_0,
              m / 64U * (n / 128U),
              64U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x64_16x8
*/
static void
__hoisted_g_gemm_bf16_64x128x64_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_16x8_0,
              m / 64U * (n / 128U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x128x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_16x16_0,
              m / 64U * (n / 128U),
              32U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_64x128x64_16x16
*/
static void
__hoisted_g_gemm_bf16_64x128x64_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 64U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(64U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(64U * (blockIdx.x / (n / 128U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_16x16_0,
              m / 64U * (n / 128U),
              32U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x32x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_8x8_0,
              m / 128U * (n / 32U),
              64U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x32_8x8
*/
static void
__hoisted_g_gemm_bf16_128x32x32_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_8x8_0,
              m / 128U * (n / 32U),
              64U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x32x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_8x16_0,
              m / 128U * (n / 32U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x32_8x16
*/
static void
__hoisted_g_gemm_bf16_128x32x32_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_8x16_0,
              m / 128U * (n / 32U),
              32U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x32x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_16x8_0,
              m / 128U * (n / 32U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x32_16x8
*/
static void
__hoisted_g_gemm_bf16_128x32x32_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_16x8_0,
              m / 128U * (n / 32U),
              32U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x32x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_16x16_0,
              m / 128U * (n / 32U),
              16U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x32_16x16
*/
static void
__hoisted_g_gemm_bf16_128x32x32_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(10240U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              10240U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_16x16_0,
              m / 128U * (n / 32U),
              16U, 10240U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x32x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_8x8_0,
              m / 128U * (n / 32U),
              64U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x64_8x8
*/
static void
__hoisted_g_gemm_bf16_128x32x64_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_8x8_0,
              m / 128U * (n / 32U),
              64U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x32x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_8x16_0,
              m / 128U * (n / 32U),
              32U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x64_8x16
*/
static void
__hoisted_g_gemm_bf16_128x32x64_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        8U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 8U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_8x16_0,
              m / 128U * (n / 32U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x32x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_16x8_0,
              m / 128U * (n / 32U),
              32U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x64_16x8
*/
static void
__hoisted_g_gemm_bf16_128x32x64_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 8U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 8U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_16x8_0,
              m / 128U * (n / 32U),
              32U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x32x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 32U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(40960U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              40960U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_16x16_0,
              m / 128U * (n / 32U),
              16U, 40960U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x32x64_16x16
*/
static void
__hoisted_g_gemm_bf16_128x32x64_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 32U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 32U)) +
                        16U * (threadIdx.x / 2U) + vrm) * n +
                       32U * (blockIdx.x % (n / 32U))
                       + 16U * (threadIdx.x % 2U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 32U)) + 16U * (threadIdx.x / 2U) +
                    vrm) * n + 32U * (blockIdx.x % (n / 32U))
                   + 16U * (threadIdx.x % 2U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(20480U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              20480U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_16x16_0,
              m / 128U * (n / 32U),
              16U, 20480U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x64x32_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_8x8_0,
              m / 128U * (n / 64U),
              128U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x32_8x8
*/
static void
__hoisted_g_gemm_bf16_128x64x32_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_8x8_0,
              m / 128U * (n / 64U),
              128U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x64x32_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_8x16_0,
              m / 128U * (n / 64U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x32_8x16
*/
static void
__hoisted_g_gemm_bf16_128x64x32_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_8x16_0,
              m / 128U * (n / 64U),
              64U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x64x32_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_16x8_0,
              m / 128U * (n / 64U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x32_16x8
*/
static void
__hoisted_g_gemm_bf16_128x64x32_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_16x8_0,
              m / 128U * (n / 64U),
              64U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x64x32_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_16x16_0,
              m / 128U * (n / 64U),
              32U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x32_16x16
*/
static void
__hoisted_g_gemm_bf16_128x64x32_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(12288U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              12288U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_16x16_0,
              m / 128U * (n / 64U),
              32U, 12288U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x64x64_8x8_0(float alpha,
                                     float beta,
                                     uint32_t n,
                                     uint32_t k,
                                     float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 float *gA,
                                                 float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_8x8_0,
              m / 128U * (n / 64U),
              128U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x64_8x8
*/
static void
__hoisted_g_gemm_bf16_128x64x64_8x8_0(__nv_bfloat16 alpha,
                                      __nv_bfloat16 beta,
                                      uint32_t n,
                                      uint32_t k,
                                      __nv_bfloat16 *gA,
                                      __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  __nv_bfloat16 *gA,
                                                  __nv_bfloat16 *gB,
                                                  __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_8x8_0,
              m / 128U * (n / 64U),
              128U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x64x64_8x16_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_8x16_0,
              m / 128U * (n / 64U),
              64U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x64_8x16
*/
static void
__hoisted_g_gemm_bf16_128x64x64_8x16_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        8U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 8U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_8x16_0,
              m / 128U * (n / 64U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x64x64_16x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_16x8_0,
              m / 128U * (n / 64U),
              64U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x64_16x8
*/
static void
__hoisted_g_gemm_bf16_128x64x64_16x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 8U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 8U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 8U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_16x8_0,
              m / 128U * (n / 64U),
              64U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x64x64_16x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 64U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(49152U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              49152U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_16x16_0,
              m / 128U * (n / 64U),
              32U, 49152U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x64x64_16x16
*/
static void
__hoisted_g_gemm_bf16_128x64x64_16x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 64U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 64U)) +
                        16U * (threadIdx.x / 4U) + vrm) * n +
                       64U * (blockIdx.x % (n / 64U))
                       + 16U * (threadIdx.x % 4U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 64U)) + 16U * (threadIdx.x / 4U) +
                    vrm) * n + 64U * (blockIdx.x % (n / 64U))
                   + 16U * (threadIdx.x % 4U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_16x16_0,
              m / 128U * (n / 64U),
              32U, 24576U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x128x32_8x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    8U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_8x8_0,
              m / 128U * (n / 128U),
              256U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x32_8x8
*/
static void
__hoisted_g_gemm_bf16_128x128x32_8x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    8U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_8x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_8x8_0,
              m / 128U * (n / 128U),
              256U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x128x32_8x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_8x16_0,
              m / 128U * (n / 128U),
              128U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x32_8x16
*/
static void
__hoisted_g_gemm_bf16_128x128x32_8x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_8x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_8x16_0,
              m / 128U * (n / 128U),
              128U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x128x32_16x8_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_16x8_0,
              m / 128U * (n / 128U),
              128U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x32_16x8
*/
static void
__hoisted_g_gemm_bf16_128x128x32_16x8_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_16x8(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_16x8_0,
              m / 128U * (n / 128U),
              128U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x128x32_16x16_0(float alpha,
                                        float beta,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 8U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    float *gA,
                                                    float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_16x16_0,
              m / 128U * (n / 128U),
              64U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x32_16x16
*/
static void
__hoisted_g_gemm_bf16_128x128x32_16x16_0(__nv_bfloat16 alpha,
                                         __nv_bfloat16 beta,
                                         uint32_t n,
                                         uint32_t k,
                                         __nv_bfloat16 *gA,
                                         __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 4096U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 32U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 32U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 8U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_16x16(__nv_bfloat16 alpha,
                                                     __nv_bfloat16 beta,
                                                     uint32_t m,
                                                     uint32_t n,
                                                     uint32_t k,
                                                     __nv_bfloat16 *gA,
                                                     __nv_bfloat16 *gB,
                                                     __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_16x16_0,
              m / 128U * (n / 128U),
              64U, 16384U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x128x64_8x8_0(float alpha,
                                      float beta,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    8U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_8x8_0,
              m / 128U * (n / 128U),
              256U, 65536U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x64_8x8
*/
static void
__hoisted_g_gemm_bf16_128x128x64_8x8_0(__nv_bfloat16 alpha,
                                       __nv_bfloat16 beta,
                                       uint32_t n,
                                       uint32_t k,
                                       __nv_bfloat16 *gA,
                                       __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    8U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_8x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   __nv_bfloat16 *gA,
                                                   __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_8x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_8x8_0,
              m / 128U * (n / 128U),
              256U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x128x64_8x16_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_8x16_0,
              m / 128U * (n / 128U),
              128U, 65536U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x64_8x16
*/
static void
__hoisted_g_gemm_bf16_128x128x64_8x16_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 8U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        8U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) + 8U * (threadIdx.x / 8U) +
                    vrm) * n + 128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_8x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_8x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_8x16_0,
              m / 128U * (n / 128U),
              128U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x128x64_16x8_0(float alpha,
                                       float beta,
                                       uint32_t n,
                                       uint32_t k,
                                       float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn]
                + alpha * rchProd[resIdxM * 8U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_16x8_0,
              m / 128U * (n / 128U),
              128U, 65536U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x64_16x8
*/
static void
__hoisted_g_gemm_bf16_128x128x64_16x8_0(__nv_bfloat16 alpha,
                                        __nv_bfloat16 beta,
                                        uint32_t n,
                                        uint32_t k,
                                        __nv_bfloat16 *gA,
                                        __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 8U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 16U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 8U * (threadIdx.x % 16U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 8U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 16U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 8U * (threadIdx.x % 16U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_16x8(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    __nv_bfloat16 *gA,
                                                    __nv_bfloat16 *gB,
                                                    __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_16x8_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_16x8_0,
              m / 128U * (n / 128U),
              128U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x128x64_16x16_0(float alpha,
                                        float beta,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(32768U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
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
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
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
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sB[row * 128U + col + k1] = local[k1];
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
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 8U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                =
                beta *
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn]
                + alpha * rchProd[resIdxM * 16U + resIdxN];
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    float *gA,
                                                    float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(65536U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              65536U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_16x16_0,
              m / 128U * (n / 128U),
              64U, 65536U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting g_gemm_bf16_128x128x64_16x16
*/
static void
__hoisted_g_gemm_bf16_128x128x64_16x16_0(__nv_bfloat16 alpha,
                                         __nv_bfloat16 beta,
                                         uint32_t n,
                                         uint32_t k,
                                         __nv_bfloat16 *gA,
                                         __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        __nv_bfloat16 *tileA = gA;
        uint32_t i0 = 0U;
        for (; i0 < 8192U; i0 += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i0 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i0 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                       tileA + (k * mrow * 128U + __anf03 * 64U + k * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf06 = bkIdx;
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (n * __anf06 * 64U + mcol * 128U + n * row +
                                col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
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
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] =
                        kpr_bf16add(old,
                                    kpr_bf16mul(rAcol[resIdxM],
                                                rBrow[resIdxN]));
                }
            }
        }
    }
    __nv_bfloat16 *t_tile = gC;
    uint32_t resIdxM = 0U;
    for (; resIdxM < 16U; resIdxM++) {
        uint32_t resIdxN = 0U;
        for (; resIdxN < 16U; resIdxN++) {
            uint32_t vrm = resIdxM;
            uint32_t vrn = resIdxN;
            __nv_bfloat16
                v0 =
                t_tile[(128U * (blockIdx.x / (n / 128U)) +
                        16U * (threadIdx.x / 8U) + vrm) * n +
                       128U * (blockIdx.x % (n / 128U))
                       + 16U * (threadIdx.x % 8U)
                       + vrn];
            __nv_bfloat16 v1 = rchProd[resIdxM * 16U + resIdxN];
            t_tile[(128U * (blockIdx.x / (n / 128U)) +
                    16U * (threadIdx.x / 8U) + vrm) * n +
                   128U * (blockIdx.x % (n / 128U))
                   + 16U * (threadIdx.x % 8U)
                   + vrn]
                = kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
        }
    }
}

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_16x16(__nv_bfloat16 alpha,
                                                     __nv_bfloat16 beta,
                                                     uint32_t m,
                                                     uint32_t n,
                                                     uint32_t k,
                                                     __nv_bfloat16 *gA,
                                                     __nv_bfloat16 *gB,
                                                     __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(32768U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_16x16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              32768U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_16x16_0,
              m / 128U * (n / 128U),
              64U, 32768U, s, alpha, beta, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
