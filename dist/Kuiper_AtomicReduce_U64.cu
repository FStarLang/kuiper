

#include "Kuiper_AtomicReduce_U64.h"

__global__

void Kuiper_AtomicReduce_U64_kernel(uint64_t *a, uint64_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  atomic_add_u64(r, a[bid * bdim + threadIdx_x()]);
}

uint64_t Kuiper_AtomicReduce_U64_reduce(size_t n, uint64_t *a)
{
  uint64_t r = 0ULL;
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(Kuiper_AtomicReduce_U64_kernel, n, (size_t)1U, (size_t)4U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

