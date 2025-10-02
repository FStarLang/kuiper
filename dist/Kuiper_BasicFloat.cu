

#include "Kuiper_BasicFloat.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(float_t *gr)
{
  *gr += 1.0f;
}

float_t Kuiper_BasicFloat_main(void)
{
  float_t r = 0.0f;
  float_t *gr = (float_t *)KPR_GPU_ALLOC(4U, 1U);
  MUST(cudaMemcpy(gr, &r, 4U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, 1U, 1U, 0U, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, 4U, cudaMemcpyDeviceToHost));
  float_t v = r;
  MUST(cudaFree(gr));
  return v;
}

