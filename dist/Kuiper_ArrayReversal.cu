

#include "Kuiper_ArrayReversal.h"

__global__
/**
  hoisted when extracting reverse_u64
*/
static void __hoisted_0(uint32_t size, uint64_t *a)
{
  uint32_t idx_ = size - blockIdx.x - (uint32_t)1U;
  uint64_t uu = a[blockIdx.x];
  a[blockIdx.x] = a[idx_];
  a[idx_] = uu;
}

void Kuiper_ArrayReversal_reverse_u64(uint32_t size, uint64_t *a)
{
  KPR_KCALL(__hoisted_0, size / (uint32_t)2U, (uint32_t)1U, (uint32_t)0U, size, a);
  cudaDeviceSynchronize();
}

