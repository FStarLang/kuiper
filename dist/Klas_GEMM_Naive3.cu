
#include "Klas_GEMM_Naive3.h"

__global__
/**
  hoisted when extracting g_matmul_bf16_rrr
*/
static void
__hoisted_g_matmul_bf16_rrr_0(uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              __nv_bfloat16 *gA,
                              __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        __nv_bfloat16 acc = __float2bfloat16(0.0f);
        __nv_bfloat16 c = __float2bfloat16(0.0f);
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            __nv_bfloat16 y =
                kpr_bf16mul(gA[trow * k + __anf0], gB[__anf0 * n + tcol]);
            __nv_bfloat16 yc = kpr_bf16sub(y, c);
            __nv_bfloat16 t = kpr_bf16add(acc, yc);
            c = kpr_bf16sub(kpr_bf16sub(t, acc), yc);
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void
Klas_GEMM_Naive3_g_matmul_bf16_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   __nv_bfloat16 *gA,
                                   __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_KCALL(__hoisted_g_matmul_bf16_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_g_matmul_f32_rrr_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, float *gA, float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float yc = gA[trow * k + __anf0] * gB[__anf0 * n + tcol] - c;
            float t = acc + yc;
            c = t - acc - yc;
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void
Klas_GEMM_Naive3_g_matmul_f32_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_g_matmul_f64_rrr_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, double *gA, double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double acc = 0.0;
        double c = 0.0;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            double yc = gA[trow * k + __anf0] * gB[__anf0 * n + tcol] - c;
            double t = acc + yc;
            c = t - acc - yc;
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void
Klas_GEMM_Naive3_g_matmul_f64_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_bf16_ccc
*/
static void
__hoisted_g_matmul_bf16_ccc_0(uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              __nv_bfloat16 *gA,
                              __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        __nv_bfloat16 acc = __float2bfloat16(0.0f);
        __nv_bfloat16 c = __float2bfloat16(0.0f);
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            __nv_bfloat16 y =
                kpr_bf16mul(gA[__anf0 * m + trow], gB[tcol * k + __anf0]);
            __nv_bfloat16 yc = kpr_bf16sub(y, c);
            __nv_bfloat16 t = kpr_bf16add(acc, yc);
            c = kpr_bf16sub(kpr_bf16sub(t, acc), yc);
            acc = t;
        }
        gC[tcol * m + trow] = acc;
    }
}

void
Klas_GEMM_Naive3_g_matmul_bf16_ccc(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   __nv_bfloat16 *gA,
                                   __nv_bfloat16 *gB, __nv_bfloat16 *gC)
{
    KPR_KCALL(__hoisted_g_matmul_bf16_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void
__hoisted_g_matmul_f32_ccc_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, float *gA, float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float yc = gA[__anf0 * m + trow] * gB[tcol * k + __anf0] - c;
            float t = acc + yc;
            c = t - acc - yc;
            acc = t;
        }
        gC[tcol * m + trow] = acc;
    }
}

void
Klas_GEMM_Naive3_g_matmul_f32_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f32_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void
__hoisted_g_matmul_f64_ccc_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, double *gA, double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double acc = 0.0;
        double c = 0.0;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            double yc = gA[__anf0 * m + trow] * gB[tcol * k + __anf0] - c;
            double t = acc + yc;
            c = t - acc - yc;
            acc = t;
        }
        gC[tcol * m + trow] = acc;
    }
}

void
Klas_GEMM_Naive3_g_matmul_f64_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f64_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
