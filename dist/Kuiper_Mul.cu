

#include "Kuiper_Mul.h"

__global__

void Kuiper_Mul_kernel(uint64_t *a1, uint64_t *a2, uint64_t *ar)
{
  size_t bid = blockIdx_x();
  ar[bid] = a1[bid] * a2[bid];
}

