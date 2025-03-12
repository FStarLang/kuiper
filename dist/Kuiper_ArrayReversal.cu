

#include "Kuiper_ArrayReversal.h"

__global__

static void kernel__uint64_t(size_t size, uint64_t *a)
{
  size_t idx = blockIdx_x();
  size_t idx_ = size - idx - (size_t)1U;
  uint64_t uu = a[idx];
  a[idx] = a[idx_];
  a[idx_] = uu;
}

void Kuiper_ArrayReversal_reverse__uint64_t(size_t size, uint64_t *a)
{
  KPR_KCALL(kernel__uint64_t, size / (size_t)2U, (size_t)1U, (size_t)4U, (size_t)0U, size, a);
  cudaDeviceSynchronize();
}

void
(*Kuiper_ArrayReversal_reverse_u64)(size_t x0, uint64_t *x1) =
  Kuiper_ArrayReversal_reverse__uint64_t;

