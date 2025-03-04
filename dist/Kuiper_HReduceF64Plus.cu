

#include "Kuiper_HReduceF64Plus.h"

size_t Kuiper_HReduceF64Plus_size = (size_t)1024U;

__global__

void Kuiper_HReduceF64Plus_k_reduce(size_t nth, double_t *a)
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

void Kuiper_HReduceF64Plus_reduce(size_t lena, double_t *a)
{
  KPR_KCALL(Kuiper_HReduceF64Plus_k_reduce, (size_t)1U, lena, lena, a);
}

