

#include "Kuiper_AtomicReduce_U32.h"

__global__

static void kernel(uint32_t *a, uint32_t *r)
{
  atomic_add_u32(r, a[blockIdx_x()]);
}

uint32_t Kuiper_AtomicReduce_U32_reduce(size_t n, uint32_t *a)
{
  uint32_t r = 0U;
  uint32_t *gr = (uint32_t *)KPR_GPU_ALLOC((size_t)4U);
  MUST(cudaMemcpy(gr, &r, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL(kernel, n, (size_t)1U, (size_t)4U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

