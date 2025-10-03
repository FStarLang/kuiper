
#include "Kuiper_HReduce.h"

__global__
/**
  hoisted when extracting reduce_f16_plus
*/
static void __hoisted_0(uint32_t lena, half_t *a)
{
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a[threadIdx.x] = __hadd(a[threadIdx.x], a[nextid]);
    }
}

void Kuiper_HReduce_reduce_f16_plus(uint32_t lena, half_t *a)
{
    KPR_KCALL(__hoisted_0, 1U, lena, 0U, lena, a);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting reduce_f32_plus
*/
static void __hoisted_1(uint32_t lena, float_t *a)
{
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a[threadIdx.x] += a[nextid];
    }
}

void Kuiper_HReduce_reduce_f32_plus(uint32_t lena, float_t *a)
{
    KPR_KCALL(__hoisted_1, 1U, lena, 0U, lena, a);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting reduce_f64_plus
*/
static void __hoisted_2(uint32_t lena, double_t *a)
{
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a[threadIdx.x] += a[nextid];
    }
}

void Kuiper_HReduce_reduce_f64_plus(uint32_t lena, double_t *a)
{
    KPR_KCALL(__hoisted_2, 1U, lena, 0U, lena, a);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting reduce_u32_plus
*/
static void __hoisted_3(uint32_t lena, uint32_t *a)
{
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a[threadIdx.x] += a[nextid];
    }
}

void Kuiper_HReduce_reduce_u32_plus(uint32_t lena, uint32_t *a)
{
    KPR_KCALL(__hoisted_3, 1U, lena, 0U, lena, a);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting reduce_u64_plus
*/
static void __hoisted_4(uint32_t lena, uint64_t *a)
{
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a[threadIdx.x] += a[nextid];
    }
}

void Kuiper_HReduce_reduce_u64_plus(uint32_t lena, uint64_t *a)
{
    KPR_KCALL(__hoisted_4, 1U, lena, 0U, lena, a);
    cudaDeviceSynchronize();
}
