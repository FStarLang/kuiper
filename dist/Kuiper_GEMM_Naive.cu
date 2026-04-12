
#include "Kuiper_GEMM_Naive.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void __hoisted_0(uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    float sum = 0.0f;
    for (; k1 < k; k1++)
        sum += gA[trow * k + k1] * gB[k1 * n + tcol];
    gC[trow * n + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_f32_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k, float *gA, float *gB, float *gC)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, m * n, 1U, 0U, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void __hoisted_1(uint32_t n, uint32_t k, double *gA, double *gB,
                        double *gC)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    double sum = 0.0l;
    for (; k1 < k; k1++)
        sum += gA[trow * k + k1] * gB[k1 * n + tcol];
    gC[trow * n + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_f64_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   double *gA, double *gB, double *gC)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_1, m * n, 1U, 0U, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void __hoisted_2(uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
                        uint32_t *gC)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    uint32_t sum = 0U;
    for (; k1 < k; k1++)
        sum += gA[trow * k + k1] * gB[k1 * n + tcol];
    gC[trow * n + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_u32_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_2, m * n, 1U, 0U, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void __hoisted_3(uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
                        uint64_t *gC)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    uint64_t sum = 0ULL;
    for (; k1 < k; k1++)
        sum += gA[trow * k + k1] * gB[k1 * n + tcol];
    gC[trow * n + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_u64_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_3, m * n, 1U, 0U, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
