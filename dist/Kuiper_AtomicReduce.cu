

#include "Kuiper_AtomicReduce.h"

__global__

static void kernel(uint64_t *a, uint64_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  atomic_add_u64(r, a[bid * bdim + threadIdx_x()]);
}

uint64_t Kuiper_AtomicReduce_reduce(size_t n, uint64_t *a)
{
  uint64_t r = 0ULL;
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(kernel, n, 1U, a, gr);
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

