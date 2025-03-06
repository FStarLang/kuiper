

#include "Kuiper_AtomicReduce_F64.h"

__global__

void Kuiper_AtomicReduce_F64_kernel(double_t *a, double_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  atomic_add_f64(r, a[bid * bdim + threadIdx_x()]);
}

double_t Kuiper_AtomicReduce_F64_reduce(size_t n, double_t *a)
{
  double_t r = (double_t)0.0l;
  double_t *gr = (double_t *)KPR_GPU_ALLOC((size_t)8U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(Kuiper_AtomicReduce_F64_kernel, n, 1U, a, gr);
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

