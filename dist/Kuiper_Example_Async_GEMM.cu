
#include "Kuiper_Example_Async_GEMM.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_0(float *a, float *b, float *s1)
{
    if (1024U * blockIdx.x + threadIdx.x < 1048576U) {
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < 1024U; k++) {
            uint32_t vk = k;
            sum +=
                a[(1024U * blockIdx.x + threadIdx.x) / 1024U * 1024U + vk] *
                b[vk * 1024U + (1024U * blockIdx.x + threadIdx.x) % 1024U];
        }
        s1[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_1(float *c1, float *d, float *s2)
{
    if (1024U * blockIdx.x + threadIdx.x < 1048576U) {
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < 1024U; k++) {
            uint32_t vk = k;
            sum +=
                c1[(1024U * blockIdx.x + threadIdx.x) / 1024U * 1024U + vk] *
                d[vk * 1024U + (1024U * blockIdx.x + threadIdx.x) % 1024U];
        }
        s2[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_2(float *r, float *s1, float *s2)
{
    if (1024U * blockIdx.x + threadIdx.x < 1048576U) {
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < 1024U; k++) {
            uint32_t vk = k;
            sum +=
                s1[(1024U * blockIdx.x + threadIdx.x) / 1024U * 1024U + vk] *
                s2[vk * 1024U + (1024U * blockIdx.x + threadIdx.x) % 1024U];
        }
        r[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

void Kuiper_Example_Async_GEMM_main(float *a, float *b, float *c1, float *d,
                                    float *r)
{
    float *s1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1048576U);
    KPR_KCALL(__hoisted_main_0, 1024U, 1024U, 0U, a, b, s1);
    float *s2 = (float *)KPR_GPU_ALLOC(sizeof(float), 1048576U);
    KPR_KCALL(__hoisted_main_1, 1024U, 1024U, 0U, c1, d, s2);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_main_2, 1024U, 1024U, 0U, r, s1, s2);
    MUST(cudaDeviceSynchronize());
    MUST(cudaFree(s1));
    MUST(cudaFree(s2));
}
