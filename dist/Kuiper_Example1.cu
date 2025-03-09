

#include "Kuiper_Example1.h"

__global__

void Kuiper_Example1_kernel(uint64_t *r)
{
  *r += 1ULL;
}

uint64_t Kuiper_Example1_main(void)
{
  uint64_t r = 1ULL;
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(Kuiper_Example1_kernel, (size_t)1U, (size_t)1U, (size_t)4U, (size_t)0U, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
  uint64_t v = r;
  MUST(cudaFree(gr));
  return v;
}

