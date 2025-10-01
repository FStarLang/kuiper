

#include "Kuiper_BasicFloat.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(float_t *gr)
{
  *gr += (float_t)1.0f;
}

float_t Kuiper_BasicFloat_main(void)
{
  float_t r = (float_t)0.0f;
  float_t *gr = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, (uint32_t)1U);
  MUST(cudaMemcpy(gr, &r, (uint32_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, (uint32_t)1U, (uint32_t)1U, (uint32_t)0U, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (uint32_t)4U, cudaMemcpyDeviceToHost));
  float_t v = r;
  MUST(cudaFree(gr));
  return v;
}

