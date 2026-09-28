
#include "Klas_GEMM_BlockTiling2D.h"

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x32x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x32x32_8x8
*/
static void
__hoisted_g_gemm_bf16_32x32x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_32x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x32x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_32x32x32_8x16
*/
static void
__hoisted_g_gemm_bf16_32x32x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x32x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_32x32x32_16x8
*/
static void
__hoisted_g_gemm_bf16_32x32x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_gemm_f32_32x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x32x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_gemm_bf16_32x32x32_16x16
*/
static void
__hoisted_g_gemm_bf16_32x32x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x32x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x32x64_8x8
*/
static void
__hoisted_g_gemm_bf16_32x32x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_32x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x32x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_32x32x64_8x16
*/
static void
__hoisted_g_gemm_bf16_32x32x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x32x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_32x32x64_16x8
*/
static void
__hoisted_g_gemm_bf16_32x32x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_gemm_f32_32x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x32x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 16U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(4)
/**
  hoisted when extracting g_gemm_bf16_32x32x64_16x16
*/
static void
__hoisted_g_gemm_bf16_32x32x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 32U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x64x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_32x64x32_8x8
*/
static void
__hoisted_g_gemm_bf16_32x64x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x64x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x64x32_8x16
*/
static void
__hoisted_g_gemm_bf16_32x64x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x64x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x64x32_16x8
*/
static void
__hoisted_g_gemm_bf16_32x64x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_32x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x64x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_32x64x32_16x16
*/
static void
__hoisted_g_gemm_bf16_32x64x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x64x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_32x64x64_8x8
*/
static void
__hoisted_g_gemm_bf16_32x64x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x64x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x64x64_8x16
*/
static void
__hoisted_g_gemm_bf16_32x64x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x64x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x64x64_16x8
*/
static void
__hoisted_g_gemm_bf16_32x64x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_32x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x64x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_32x64x64_16x16
*/
static void
__hoisted_g_gemm_bf16_32x64x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 32U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_32x128x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_32x128x32_8x8
*/
static void
__hoisted_g_gemm_bf16_32x128x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_32x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_32x128x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_32x128x32_8x16
*/
static void
__hoisted_g_gemm_bf16_32x128x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_32x128x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_32x128x32_16x8
*/
static void
__hoisted_g_gemm_bf16_32x128x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_32x128x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x128x32_16x16
*/
static void
__hoisted_g_gemm_bf16_32x128x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_32x128x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_32x128x64_8x8
*/
static void
__hoisted_g_gemm_bf16_32x128x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_32x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_32x128x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_32x128x64_8x16
*/
static void
__hoisted_g_gemm_bf16_32x128x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_32x128x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_32x128x64_16x8
*/
static void
__hoisted_g_gemm_bf16_32x128x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_32x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_32x128x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_32x128x64_16x16
*/
static void
__hoisted_g_gemm_bf16_32x128x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 32U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 32U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 32U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 32U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x32x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x32x32_8x8
*/
static void
__hoisted_g_gemm_bf16_64x32x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_64x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x32x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_64x32x32_8x16
*/
static void
__hoisted_g_gemm_bf16_64x32x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x32x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_64x32x32_16x8
*/
static void
__hoisted_g_gemm_bf16_64x32x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_64x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x32x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_64x32x32_16x16
*/
static void
__hoisted_g_gemm_bf16_64x32x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x32x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x32x64_8x8
*/
static void
__hoisted_g_gemm_bf16_64x32x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_64x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x32x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_64x32x64_8x16
*/
static void
__hoisted_g_gemm_bf16_64x32x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x32x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_64x32x64_16x8
*/
static void
__hoisted_g_gemm_bf16_64x32x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_f32_64x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x32x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 32U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(8)
/**
  hoisted when extracting g_gemm_bf16_64x32x64_16x16
*/
static void
__hoisted_g_gemm_bf16_64x32x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 64U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x64x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_64x64x32_8x8
*/
static void
__hoisted_g_gemm_bf16_64x64x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x64x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x64x32_8x16
*/
static void
__hoisted_g_gemm_bf16_64x64x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x64x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x64x32_16x8
*/
static void
__hoisted_g_gemm_bf16_64x64x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_64x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x64x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_64x64x32_16x16
*/
static void
__hoisted_g_gemm_bf16_64x64x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x64x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_64x64x64_8x8
*/
static void
__hoisted_g_gemm_bf16_64x64x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x64x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x64x64_8x16
*/
static void
__hoisted_g_gemm_bf16_64x64x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x64x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x64x64_16x8
*/
static void
__hoisted_g_gemm_bf16_64x64x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_64x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x64x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_64x64x64_16x16
*/
static void
__hoisted_g_gemm_bf16_64x64x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 64U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_64x128x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_64x128x32_8x8
*/
static void
__hoisted_g_gemm_bf16_64x128x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_64x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_64x128x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_64x128x32_8x16
*/
static void
__hoisted_g_gemm_bf16_64x128x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_64x128x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_64x128x32_16x8
*/
static void
__hoisted_g_gemm_bf16_64x128x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_64x128x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x128x32_16x16
*/
static void
__hoisted_g_gemm_bf16_64x128x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 2048U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_64x128x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_64x128x64_8x8
*/
static void
__hoisted_g_gemm_bf16_64x128x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_64x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_64x128x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_64x128x64_8x16
*/
static void
__hoisted_g_gemm_bf16_64x128x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_64x128x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_64x128x64_16x8
*/
static void
__hoisted_g_gemm_bf16_64x128x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_64x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_64x128x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_64x128x64_16x16
*/
static void
__hoisted_g_gemm_bf16_64x128x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 64U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 64U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 64U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 64U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x32x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x32x32_8x8
*/
static void
__hoisted_g_gemm_bf16_128x32x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_128x32x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x32x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_128x32x32_8x16
*/
static void
__hoisted_g_gemm_bf16_128x32x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x32x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_128x32x32_16x8
*/
static void
__hoisted_g_gemm_bf16_128x32x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_128x32x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x32x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_128x32x32_16x16
*/
static void
__hoisted_g_gemm_bf16_128x32x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x32x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x32x64_8x8
*/
static void
__hoisted_g_gemm_bf16_128x32x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_128x32x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x32x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_128x32x64_8x16
*/
static void
__hoisted_g_gemm_bf16_128x32x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x32x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_128x32x64_16x8
*/
static void
__hoisted_g_gemm_bf16_128x32x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 8U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 4U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 4U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_128x32x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x32x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 64U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_bf16_128x32x64_16x16
*/
static void
__hoisted_g_gemm_bf16_128x32x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 32U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 32U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 2U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 32U + 16U * (threadIdx.x % 2U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 32U) * 128U + threadIdx.x / 2U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 32U) * 32U + threadIdx.x % 2U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x64x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_128x64x32_8x8
*/
static void
__hoisted_g_gemm_bf16_128x64x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x64x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x64x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x64x32_8x16
*/
static void
__hoisted_g_gemm_bf16_128x64x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x64x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x64x32_16x8
*/
static void
__hoisted_g_gemm_bf16_128x64x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_128x64x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x64x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_128x64x32_16x16
*/
static void
__hoisted_g_gemm_bf16_128x64x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 2048U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x64x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_128x64x64_8x8
*/
static void
__hoisted_g_gemm_bf16_128x64x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x64x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x64x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x64x64_8x16
*/
static void
__hoisted_g_gemm_bf16_128x64x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x64x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x64x64_16x8
*/
static void
__hoisted_g_gemm_bf16_128x64x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 8U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 8U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 8U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_128x64x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x64x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 128U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_bf16_128x64x64_16x16
*/
static void
__hoisted_g_gemm_bf16_128x64x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 64U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 64U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 4U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 64U + 16U * (threadIdx.x % 4U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 64U) * 128U + threadIdx.x / 4U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 64U) * 64U + threadIdx.x % 4U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x8
*/
static void
__hoisted_g_gemm_f32_128x128x32_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_bf16_128x128x32_8x8
*/
static void
__hoisted_g_gemm_bf16_128x128x32_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_128x128x32_8x16
*/
static void
__hoisted_g_gemm_f32_128x128x32_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_128x128x32_8x16
*/
static void
__hoisted_g_gemm_bf16_128x128x32_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x8
*/
static void
__hoisted_g_gemm_f32_128x128x32_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] =
                    sarA[dotIdx * 128U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_128x128x32_16x8
*/
static void
__hoisted_g_gemm_bf16_128x128x32_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] =
                    sarA[dotIdx * 128U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x128x32_16x16
*/
static void
__hoisted_g_gemm_f32_128x128x32_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 32U;
            uint32_t col = (i + threadIdx.x * 4U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x128x32_16x16
*/
static void
__hoisted_g_gemm_bf16_128x128x32_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    uint32_t num_k_tiles = k / 32U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 32U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 32U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 32U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x8
*/
static void
__hoisted_g_gemm_f32_128x128x64_8x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[64U];
    memset(rchProd, 0U, 64U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_bf16_128x128x64_8x8
*/
static void
__hoisted_g_gemm_bf16_128x128x64_8x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 64U);
    __nv_bfloat16 rchProd[64U];
    for (uint32_t _i = 0U; _i < 64U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 2048U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 64U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 8U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_128x128x64_8x16
*/
static void
__hoisted_g_gemm_f32_128x128x64_8x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_128x128x64_8x16
*/
static void
__hoisted_g_gemm_bf16_128x128x64_8x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rAcol[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 8U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 8U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 8U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x8
*/
static void
__hoisted_g_gemm_f32_128x128x64_16x8_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[128U];
    memset(rchProd, 0U, 128U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] =
                    sarA[dotIdx * 128U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(128)
/**
  hoisted when extracting g_gemm_bf16_128x128x64_16x8
*/
static void
__hoisted_g_gemm_bf16_128x128x64_16x8_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 128U);
    __nv_bfloat16 rchProd[128U];
    for (uint32_t _i = 0U; _i < 128U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 1024U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 rBrow[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] =
                    sarA[dotIdx * 128U + 16U * (threadIdx.x / 16U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 8U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 8U * (threadIdx.x % 16U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 8U; resIdxN++) {
                    uint32_t idx = resIdxM * 8U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 128U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 16U * 16U + ci0 / 8U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 16U * 8U + ci0 % 8U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_f32_128x128x64_16x16
*/
static void
__hoisted_g_gemm_f32_128x128x64_16x16_0(float *gA, float *gB, uint32_t k,
    uint32_t n, float *gC, float beta, float alpha)
{
    float rchProd[256U];
    memset(rchProd, 0U, 256U * sizeof(float));
    float *sarA = (float *) KPR_SHMEM_AT(0U);
    float *sarB = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i + threadIdx.x * 4U) / 64U;
            uint32_t col = (i + threadIdx.x * 4U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 256U) {
            float local[4U];
            memset(local, 0U, 4U * sizeof(float));
            uint32_t row = (i1 + threadIdx.x * 4U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 4U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 4U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
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
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
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
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * rchProd[ci0];
    }
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting g_gemm_bf16_128x128x64_16x16
*/
static void
__hoisted_g_gemm_bf16_128x128x64_16x16_0(__nv_bfloat16 *gA, __nv_bfloat16 *gB,
    uint32_t k, uint32_t n, __nv_bfloat16 *gC, __nv_bfloat16 beta,
    __nv_bfloat16 alpha)
{
    KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 256U);
    __nv_bfloat16 rchProd[256U];
    for (uint32_t _i = 0U; _i < 256U; ++_i)
        rchProd[_i] = __float2bfloat16(0.0f);
    __nv_bfloat16 *sarA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sarB = (__nv_bfloat16 *) KPR_SHMEM_AT(
        (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    uint32_t num_k_tiles = k / 64U;
    uint32_t num_n_tiles = n / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 8192U; i += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(
                local, gA + (k * mrow * 128U + __anf0 * 64U + k * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarA[(col + k1) * 128U + row] = local[k1];
        }
        uint32_t __anf01 = bkIdx;
        uint32_t i1 = 0U;
        for (; i1 < 8192U; i1 += 512U) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 8U);
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 128U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 128U;
            vec_memcpy(
                local, gB + (n * __anf01 * 64U + mcol * 128U + n * row + col));
            uint32_t k1 = 0U;
            for (; k1 < 8U; k1++)
                sarB[row * 128U + col + k1] = local[k1];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 64U; dotIdx++) {
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rAcol[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rAcol[_i] = __float2bfloat16(0.0f);
            KRML_CHECK_SIZE(sizeof(__nv_bfloat16), 16U);
            __nv_bfloat16 rBrow[16U];
            for (uint32_t _i = 0U; _i < 16U; ++_i)
                rBrow[_i] = __float2bfloat16(0.0f);
            uint32_t j0 = 0U;
            for (; j0 < 16U; j0++)
                rAcol[j0] = sarA[dotIdx * 128U + 16U * (threadIdx.x / 8U) + j0];
            uint32_t j1 = 0U;
            for (; j1 < 16U; j1++)
                rBrow[j1] = sarB[dotIdx * 128U + 16U * (threadIdx.x % 8U) + j1];
            uint32_t resIdxM = 0U;
            for (; resIdxM < 16U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 16U; resIdxN++) {
                    uint32_t idx = resIdxM * 16U + resIdxN;
                    __nv_bfloat16 old = rchProd[idx];
                    rchProd[idx] = kpr_bf16add(
                        old, kpr_bf16mul(rAcol[resIdxM], rBrow[resIdxN]));
                }
            }
        }
    }
    uint32_t cctr = 0U;
    for (; cctr < 256U; cctr++) {
        uint32_t ci0 = cctr;
        uint32_t grow_sz =
            blockIdx.x / (n / 128U) * 128U + threadIdx.x / 8U * 16U + ci0 / 16U;
        uint32_t gcol_sz =
            blockIdx.x % (n / 128U) * 128U + threadIdx.x % 8U * 16U + ci0 % 16U;
        __nv_bfloat16 v0 = gC[grow_sz * n + gcol_sz];
        __nv_bfloat16 v1 = rchProd[ci0];
        gC[grow_sz * n + gcol_sz] =
            kpr_bf16add(kpr_bf16mul(beta, v0), kpr_bf16mul(alpha, v1));
    }
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_8x8_0, m / 32U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_8x8_0, m / 32U * (n / 32U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_8x16_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_8x16_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_16x8_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_16x8_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x32_16x16_0, m / 32U * (n / 32U), 4U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x32_16x16_0, m / 32U * (n / 32U), 4U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_8x8_0, m / 32U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_8x8_0, m / 32U * (n / 32U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_8x16_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_8x16_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_16x8_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_16x8_0, m / 32U * (n / 32U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x32x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x32x64_16x16_0, m / 32U * (n / 32U), 4U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x32x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x32x64_16x16_0, m / 32U * (n / 32U), 4U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_8x8_0, m / 32U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_8x8_0, m / 32U * (n / 64U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_8x16_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_8x16_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_16x8_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_16x8_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x32_16x16_0, m / 32U * (n / 64U), 8U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x32_16x16_0, m / 32U * (n / 64U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_8x8_0, m / 32U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_8x8_0, m / 32U * (n / 64U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_8x16_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_8x16_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_16x8_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_16x8_0, m / 32U * (n / 64U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x64x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x64x64_16x16_0, m / 32U * (n / 64U), 8U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x64x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x64x64_16x16_0, m / 32U * (n / 64U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_8x8_0, m / 32U * (n / 128U), 64U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_8x8_0, m / 32U * (n / 128U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_8x16_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_8x16_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_16x8_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_16x8_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x32_16x16_0, m / 32U * (n / 128U), 16U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 1024U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 1024U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x32_16x16_0, m / 32U * (n / 128U),
        16U,
        (uint32_t) sizeof(__nv_bfloat16) * 1024U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_8x8_0, m / 32U * (n / 128U), 64U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_8x8_0, m / 32U * (n / 128U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_8x16_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_8x16_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_16x8_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_16x8_0, m / 32U * (n / 128U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_32x128x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_32x128x64_16x16_0, m / 32U * (n / 128U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 32U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_32x128x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_32x128x64_16x16_0, m / 32U * (n / 128U),
        16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_8x8_0, m / 64U * (n / 32U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_8x8_0, m / 64U * (n / 32U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_8x16_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_8x16_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_16x8_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_16x8_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x32_16x16_0, m / 64U * (n / 32U), 8U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x32_16x16_0, m / 64U * (n / 32U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_8x8_0, m / 64U * (n / 32U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_8x8_0, m / 64U * (n / 32U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_8x16_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_8x16_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_16x8_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_16x8_0, m / 64U * (n / 32U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x32x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x32x64_16x16_0, m / 64U * (n / 32U), 8U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x32x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x32x64_16x16_0, m / 64U * (n / 32U), 8U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_8x8_0, m / 64U * (n / 64U), 64U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_8x8_0, m / 64U * (n / 64U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_8x16_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_8x16_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_16x8_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_16x8_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x32_16x16_0, m / 64U * (n / 64U), 16U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x32_16x16_0, m / 64U * (n / 64U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_8x8_0, m / 64U * (n / 64U), 64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_8x8_0, m / 64U * (n / 64U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_8x16_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_8x16_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_16x8_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_16x8_0, m / 64U * (n / 64U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x64x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x64x64_16x16_0, m / 64U * (n / 64U), 16U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x64x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x64x64_16x16_0, m / 64U * (n / 64U), 16U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_8x8_0, m / 64U * (n / 128U), 128U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_8x8_0, m / 64U * (n / 128U), 128U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_8x16_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_8x16_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_16x8_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_16x8_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 2048U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x32_16x16_0, m / 64U * (n / 128U), 32U,
        (uint32_t) sizeof(float) * 2048U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 2048U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 2048U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x32_16x16_0, m / 64U * (n / 128U),
        32U,
        (uint32_t) sizeof(__nv_bfloat16) * 2048U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_8x8_0, m / 64U * (n / 128U), 128U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_8x8_0, m / 64U * (n / 128U), 128U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_8x16_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_8x16_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_16x8_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_16x8_0, m / 64U * (n / 128U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_64x128x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_64x128x64_16x16_0, m / 64U * (n / 128U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 64U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_64x128x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_64x128x64_16x16_0, m / 64U * (n / 128U),
        32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_8x8_0, m / 128U * (n / 32U), 64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_8x8_0, m / 128U * (n / 32U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_8x16_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_8x16_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_16x8_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_16x8_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x32_16x16_0, m / 128U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 1024U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x32_16x16_0, m / 128U * (n / 32U),
        16U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 1024U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_8x8_0, m / 128U * (n / 32U), 64U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_8x8_0, m / 128U * (n / 32U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_8x16_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_8x16_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_16x8_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_16x8_0, m / 128U * (n / 32U), 32U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x32x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x32x64_16x16_0, m / 128U * (n / 32U), 16U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 32U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x32x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x32x64_16x16_0, m / 128U * (n / 32U),
        16U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_8x8_0, m / 128U * (n / 64U), 128U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_8x8_0, m / 128U * (n / 64U), 128U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_8x16_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_8x16_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_16x8_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_16x8_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x32_16x16_0, m / 128U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 2048U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 2048U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 2048U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x32_16x16_0, m / 128U * (n / 64U),
        32U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 2048U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_8x8_0, m / 128U * (n / 64U), 128U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_8x8_0, m / 128U * (n / 64U), 128U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_8x16_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_8x16_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_16x8_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_16x8_0, m / 128U * (n / 64U), 64U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x64x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x64x64_16x16_0, m / 128U * (n / 64U), 32U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 64U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x64x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x64x64_16x16_0, m / 128U * (n / 64U),
        32U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_8x8_0, m / 128U * (n / 128U),
        256U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_8x8_0, m / 128U * (n / 128U),
        256U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_8x16_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_8x16_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_16x8_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_16x8_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x16(float alpha,
    float beta, uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB,
    float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U);
    if ((uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 4096U +
                (uint32_t) sizeof(float) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x32_16x16_0, m / 128U * (n / 128U),
        64U,
        (uint32_t) sizeof(float) * 4096U + (uint32_t) sizeof(float) * 4096U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 32U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 4096U +
                   (uint32_t) sizeof(__nv_bfloat16) * 4096U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x32_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 4096U +
                (uint32_t) sizeof(__nv_bfloat16) * 4096U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x32_16x16_0, m / 128U * (n / 128U),
        64U,
        (uint32_t) sizeof(__nv_bfloat16) * 4096U +
            (uint32_t) sizeof(__nv_bfloat16) * 4096U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_8x8_0, m / 128U * (n / 128U),
        256U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_8x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_8x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_8x8_0, m / 128U * (n / 128U),
        256U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x16(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_8x16_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_8x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_8x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_8x16_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x8(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_16x8_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_16x8(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_16x8_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_16x8_0, m / 128U * (n / 128U),
        128U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x16(float alpha,
    float beta, uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB,
    float *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U);
    if ((uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_128x128x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 8192U +
                (uint32_t) sizeof(float) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_128x128x64_16x16_0, m / 128U * (n / 128U),
        64U,
        (uint32_t) sizeof(float) * 8192U + (uint32_t) sizeof(float) * 8192U, s,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_16x16(__nv_bfloat16 alpha,
    __nv_bfloat16 beta, uint32_t m, uint32_t n, uint32_t k, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_GUARD(m % 128U == 0U);
    KPR_GUARD(k % 64U == 0U);
    KPR_GUARD(n % 128U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(__nv_bfloat16) * 8192U +
                   (uint32_t) sizeof(__nv_bfloat16) * 8192U);
    if ((uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_bf16_128x128x64_16x16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(__nv_bfloat16) * 8192U +
                (uint32_t) sizeof(__nv_bfloat16) * 8192U));
    KPR_KCALL(__hoisted_g_gemm_bf16_128x128x64_16x16_0, m / 128U * (n / 128U),
        64U,
        (uint32_t) sizeof(__nv_bfloat16) * 8192U +
            (uint32_t) sizeof(__nv_bfloat16) * 8192U,
        s, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
