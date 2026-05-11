
#include "Klas_GEMM_Naive3.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void __hoisted_0(uint32_t m, uint32_t n, uint32_t k, float *gA,
                        float *gB, float *gC)
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
    KPR_KCALL(__hoisted_0, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void __hoisted_1(uint32_t m, uint32_t n, uint32_t k, double *gA,
                        double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double acc = 0.0l;
        double c = 0.0l;
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
    KPR_KCALL(__hoisted_1, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void __hoisted_2(uint32_t m, uint32_t n, uint32_t k, float *gA,
                        float *gB, float *gC)
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
    KPR_KCALL(__hoisted_2, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void __hoisted_3(uint32_t m, uint32_t n, uint32_t k, double *gA,
                        double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double acc = 0.0l;
        double c = 0.0l;
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
    KPR_KCALL(__hoisted_3, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}
