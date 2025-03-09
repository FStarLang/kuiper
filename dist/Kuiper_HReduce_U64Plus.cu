

#include "Kuiper_HReduce_U64Plus.h"

size_t Kuiper_HReduce_U64Plus_size = (size_t)1024U;

__global__

static void k_reduce(size_t nth, uint64_t *a)
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
        a[tid] += a[nextid];
    n = it + (size_t)1U;
  }
}

void Kuiper_HReduce_U64Plus_reduce(size_t lena, uint64_t *a)
{
  KPR_KCALL(k_reduce, (size_t)1U, lena, (size_t)4U, (size_t)0U, lena, a);
  cudaDeviceSynchronize();
}

