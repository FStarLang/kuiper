

#include "Kuiper_HReduce_F16Plus.h"

size_t Kuiper_HReduce_F16Plus_size = (size_t)1024U;

__device__

static void k_reduce(size_t nth, half_t *a)
{
  size_t tid = threadIdx_x();
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < nth)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = tid + (size_t)(1U << (uint32_t)it);
    if (nextid < nth)
      if ((tid & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        a[tid] = __hadd(a[tid], a[nextid]);
    n = it + (size_t)1U;
  }
}

__global__

static void __hoisted_0(size_t lena, half_t *a)
{
  k_reduce(lena, a);
}

void Kuiper_HReduce_F16Plus_reduce(size_t lena, half_t *a)
{
  KPR_KCALL(__hoisted_0, (size_t)1U, lena, (size_t)4U, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

