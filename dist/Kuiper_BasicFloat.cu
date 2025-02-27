

#include "Kuiper_BasicFloat.h"

__global__

void Kuiper_BasicFloat_kernel(float_t *r)
{
  *r += (float_t)1.0f;
}

float_t Kuiper_BasicFloat_main(void)
{
  float_t r = (float_t)0.0f;
  float_t *gr = (float_t *)KPR_GPU_ALLOC((size_t)4U);
  MUST(cudaMemcpy(gr, &r, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL_ASYNC(Kuiper_BasicFloat_kernel, 1U, 1U, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)4U, cudaMemcpyDeviceToHost));
  float_t v = r;
  MUST(cudaFree(gr));
  return v;
}

