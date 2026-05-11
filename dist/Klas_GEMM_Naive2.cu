
#include "Klas_GEMM_Naive2.h"

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
        float sum = 0.0f;
        for (; k1 < k; k1++)
            sum += gA[trow * k + k1] * gB[k1 * n + tcol];
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f32_rrr(uint32_t m,
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
        double sum = 0.0l;
        for (; k1 < k; k1++)
            sum += gA[trow * k + k1] * gB[k1 * n + tcol];
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f64_rrr(uint32_t m,
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
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_2(uint32_t m, uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
            uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint32_t sum = 0U;
        for (; k1 < k; k1++)
            sum += gA[trow * k + k1] * gB[k1 * n + tcol];
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u32_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_KCALL(__hoisted_2, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_3(uint32_t m, uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
            uint64_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint64_t sum = 0ULL;
        for (; k1 < k; k1++)
            sum += gA[trow * k + k1] * gB[k1 * n + tcol];
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u64_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_KCALL(__hoisted_3, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void __hoisted_4(uint32_t m, uint32_t n, uint32_t k, float *gA,
                        float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float sum = 0.0f;
        for (; k1 < k; k1++)
            sum += gA[k1 * m + trow] * gB[tcol * k + k1];
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f32_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_KCALL(__hoisted_4, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void __hoisted_5(uint32_t m, uint32_t n, uint32_t k, double *gA,
                        double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double sum = 0.0l;
        for (; k1 < k; k1++)
            sum += gA[k1 * m + trow] * gB[tcol * k + k1];
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f64_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC)
{
    KPR_KCALL(__hoisted_5, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_ccc
*/
static void
__hoisted_6(uint32_t m, uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
            uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint32_t sum = 0U;
        for (; k1 < k; k1++)
            sum += gA[k1 * m + trow] * gB[tcol * k + k1];
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u32_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_KCALL(__hoisted_6, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_ccc
*/
static void
__hoisted_7(uint32_t m, uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
            uint64_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint64_t sum = 0ULL;
        for (; k1 < k; k1++)
            sum += gA[k1 * m + trow] * gB[tcol * k + k1];
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u64_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_KCALL(__hoisted_7, (m * n + 1023U) / 1024U, 1024U, 0U, m, n, k, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}
