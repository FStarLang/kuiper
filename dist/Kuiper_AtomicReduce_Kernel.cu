

#include "Kuiper_AtomicReduce_Kernel.h"

__global__

void Kuiper_AtomicReduce_Kernel_kernel(uint64_t *a, uint64_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  atomic_add_u64(r, a[bid * bdim + threadIdx_x()]);
}

