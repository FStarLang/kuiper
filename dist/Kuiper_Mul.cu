

#include "Kuiper_Mul.h"

__global__

void Kuiper_Mul_kernel(uint64_t *a1, uint64_t *a2, uint64_t *ar)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t tid = bid * bdim + threadIdx_x();
  ar[tid] = a1[tid] * a2[tid];
}

