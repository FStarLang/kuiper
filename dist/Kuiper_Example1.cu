

#include "Kuiper_Example1.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(uint64_t *gr)
{
  *gr += 1ULL;
}

uint64_t Kuiper_Example1_main(void)
{
  uint64_t r = 1ULL;
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, (uint32_t)1U);
  MUST(cudaMemcpy(gr, &r, (uint32_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, (uint32_t)1U, (uint32_t)1U, (uint32_t)0U, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (uint32_t)8U, cudaMemcpyDeviceToHost));
  uint64_t v = r;
  MUST(cudaFree(gr));
  return v;
}

