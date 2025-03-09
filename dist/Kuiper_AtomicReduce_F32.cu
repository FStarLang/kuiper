

#include "Kuiper_AtomicReduce_F32.h"

__global__

static void kernel(float_t *a, float_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  atomic_add_f32(r, a[bid * bdim + threadIdx_x()]);
}

float_t Kuiper_AtomicReduce_F32_reduce(size_t n, float_t *a)
{
  float_t r = (float_t)0.0f;
  float_t *gr = (float_t *)KPR_GPU_ALLOC((size_t)4U);
  MUST(cudaMemcpy(gr, &r, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL(kernel, n, (size_t)1U, (size_t)4U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

