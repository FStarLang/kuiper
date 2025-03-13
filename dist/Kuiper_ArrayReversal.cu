

#include "Kuiper_ArrayReversal.h"

__global__

static void __hoisted_0(size_t size, uint64_t *a)
{
  size_t idx = blockIdx_x();
  size_t idx_ = size - idx - (size_t)1U;
  uint64_t uu = a[idx];
  a[idx] = a[idx_];
  a[idx_] = uu;
}

void Kuiper_ArrayReversal_reverse_u64(size_t size, uint64_t *a)
{
  KPR_KCALL(__hoisted_0, size / (size_t)2U, (size_t)1U, (size_t)4U, (size_t)0U, size, a);
  cudaDeviceSynchronize();
}

