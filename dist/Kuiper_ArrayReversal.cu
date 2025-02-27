

#include "Kuiper_ArrayReversal.h"

__global__

void Kuiper_ArrayReversal_kernel__uint64_t(size_t size, uint64_t *a)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t idx = bid * bdim + threadIdx_x();
  size_t idx_ = size - idx - (size_t)1U;
  uint64_t uu = a[idx];
  a[idx] = a[idx_];
  a[idx_] = uu;
}

void Kuiper_ArrayReversal_reverse__uint64_t(size_t size, uint64_t *a)
{
  KPR_KCALL(Kuiper_ArrayReversal_kernel__uint64_t, size / (size_t)2U, 1U, size, a);
}

void
(*Kuiper_ArrayReversal_reverse_u64)(size_t x0, uint64_t *x1) =
  Kuiper_ArrayReversal_reverse__uint64_t;

