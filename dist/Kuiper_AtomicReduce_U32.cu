

#include "Kuiper_AtomicReduce_U32.h"

__global__

void Kuiper_AtomicReduce_U32_kernel(uint32_t *a, uint32_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  atomic_add_u32(r, a[bid * bdim + threadIdx_x()]);
}

uint32_t Kuiper_AtomicReduce_U32_reduce(size_t n, uint32_t *a)
{
  uint32_t r = 0U;
  uint32_t *gr = (uint32_t *)KPR_GPU_ALLOC((size_t)4U);
  MUST(cudaMemcpy(gr, &r, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL(Kuiper_AtomicReduce_U32_kernel, n, 1U, a, gr);
  MUST(cudaMemcpy(&r, gr, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

