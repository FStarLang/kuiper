
#include "Kuiper_Example_Async_GEMM.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_0(float *a, float *b, float *s1)
{
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < 1024U; k++)
        sum +=
            a[blockIdx.x / 1024U * 1024U + k] * b[k * 1024U +
                                                  blockIdx.x % 1024U];
    s1[blockIdx.x] = sum;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_1(float *c1, float *d, float *s2)
{
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < 1024U; k++)
        sum +=
            c1[blockIdx.x / 1024U * 1024U + k] * d[k * 1024U +
                                                   blockIdx.x % 1024U];
    s2[blockIdx.x] = sum;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_2(float *r, float *s1, float *s2)
{
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < 1024U; k++)
        sum +=
            s1[blockIdx.x / 1024U * 1024U + k] * s2[k * 1024U +
                                                    blockIdx.x % 1024U];
    r[blockIdx.x] = sum;
}

void Kuiper_Example_Async_GEMM_main(float *a, float *b, float *c1, float *d,
                                    float *r)
{
    float *s1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1048576U);
    KPR_KCALL(__hoisted_main_0, 1048576U, 1U, 0U, a, b, s1);
    float *s2 = (float *)KPR_GPU_ALLOC(sizeof(float), 1048576U);
    KPR_KCALL(__hoisted_main_1, 1048576U, 1U, 0U, c1, d, s2);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_main_2, 1048576U, 1U, 0U, r, s1, s2);
    MUST(cudaDeviceSynchronize());
    MUST(cudaFree(s1));
    MUST(cudaFree(s2));
}
