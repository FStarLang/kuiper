

#include "Kuiper_HReduce.h"

__global__

/**
  hoisted when extracting reduce_f16_plus
*/
static void __hoisted_0(size_t lena, half_t *a)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t __anf14641 = n;
    __syncthreads();
    size_t nextid = threadIdx_x + (size_t)(1U << (uint32_t)__anf14641);
    if (nextid < lena)
      if
      (
        (threadIdx_x & (size_t)(1U << (uint32_t)(__anf14641 + (size_t)1U)) - (size_t)1U) ==
          (size_t)0U
      )
        a[threadIdx_x] = __hadd(a[threadIdx_x], a[nextid]);
    n += (size_t)1U;
  }
}

void Kuiper_HReduce_reduce_f16_plus(size_t lena, half_t *a)
{
  KPR_KCALL(__hoisted_0, (size_t)1U, lena, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

__global__

/**
  hoisted when extracting reduce_f32_plus
*/
static void __hoisted_1(size_t lena, float_t *a)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t __anf14641 = n;
    __syncthreads();
    size_t nextid = threadIdx_x + (size_t)(1U << (uint32_t)__anf14641);
    if (nextid < lena)
      if
      (
        (threadIdx_x & (size_t)(1U << (uint32_t)(__anf14641 + (size_t)1U)) - (size_t)1U) ==
          (size_t)0U
      )
        a[threadIdx_x] += a[nextid];
    n += (size_t)1U;
  }
}

void Kuiper_HReduce_reduce_f32_plus(size_t lena, float_t *a)
{
  KPR_KCALL(__hoisted_1, (size_t)1U, lena, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

__global__

/**
  hoisted when extracting reduce_f64_plus
*/
static void __hoisted_2(size_t lena, double_t *a)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t __anf14641 = n;
    __syncthreads();
    size_t nextid = threadIdx_x + (size_t)(1U << (uint32_t)__anf14641);
    if (nextid < lena)
      if
      (
        (threadIdx_x & (size_t)(1U << (uint32_t)(__anf14641 + (size_t)1U)) - (size_t)1U) ==
          (size_t)0U
      )
        a[threadIdx_x] += a[nextid];
    n += (size_t)1U;
  }
}

void Kuiper_HReduce_reduce_f64_plus(size_t lena, double_t *a)
{
  KPR_KCALL(__hoisted_2, (size_t)1U, lena, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

__global__

/**
  hoisted when extracting reduce_u32_plus
*/
static void __hoisted_3(size_t lena, uint32_t *a)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t __anf14641 = n;
    __syncthreads();
    size_t nextid = threadIdx_x + (size_t)(1U << (uint32_t)__anf14641);
    if (nextid < lena)
      if
      (
        (threadIdx_x & (size_t)(1U << (uint32_t)(__anf14641 + (size_t)1U)) - (size_t)1U) ==
          (size_t)0U
      )
        a[threadIdx_x] += a[nextid];
    n += (size_t)1U;
  }
}

void Kuiper_HReduce_reduce_u32_plus(size_t lena, uint32_t *a)
{
  KPR_KCALL(__hoisted_3, (size_t)1U, lena, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

__global__

/**
  hoisted when extracting reduce_u64_plus
*/
static void __hoisted_4(size_t lena, uint64_t *a)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t __anf14641 = n;
    __syncthreads();
    size_t nextid = threadIdx_x + (size_t)(1U << (uint32_t)__anf14641);
    if (nextid < lena)
      if
      (
        (threadIdx_x & (size_t)(1U << (uint32_t)(__anf14641 + (size_t)1U)) - (size_t)1U) ==
          (size_t)0U
      )
        a[threadIdx_x] += a[nextid];
    n += (size_t)1U;
  }
}

void Kuiper_HReduce_reduce_u64_plus(size_t lena, uint64_t *a)
{
  KPR_KCALL(__hoisted_4, (size_t)1U, lena, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

