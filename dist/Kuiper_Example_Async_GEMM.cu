
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
    cudaStream_t str1 = KPR_FRESH_STREAM();
    cudaStream_t str2 = KPR_FRESH_STREAM();
    float *s1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1048576U);
    KPR_KCALL(__hoisted_main_0, 1024U, 1024U, 0U, str1, a, b, s1);
    float *s2 = (float *)KPR_GPU_ALLOC(sizeof(float), 1048576U);
    KPR_KCALL(__hoisted_main_1, 1024U, 1024U, 0U, str2, c1, d, s2);
    MUST(cudaStreamSynchronize(str1));
    MUST(cudaStreamSynchronize(str2));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_main_2, 1024U, 1024U, 0U, s, r, s1, s2);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaFree(s1));
    MUST(cudaFree(s2));
    MUST(cudaStreamDestroy(str1));
    MUST(cudaStreamDestroy(str2));
}
