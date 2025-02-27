

#include "Kuiper_Polymorphism1.h"

void Kuiper_Polymorphism1_kswap__uint64_t(uint64_t *r1, uint64_t *r2)
{
  uint64_t v1 = *r1;
  *r1 = *r2;
  *r2 = v1;
}

__global__

void Kuiper_Polymorphism1_kswap_U64(uint64_t *r1, uint64_t *r2)
{
  Kuiper_Polymorphism1_kswap__uint64_t(r1, r2);
}

void Kuiper_Polymorphism1_kswap__float_t(float_t *r1, float_t *r2)
{
  float_t v1 = *r1;
  *r1 = *r2;
  *r2 = v1;
}

__global__

void Kuiper_Polymorphism1_kswap_F32(float_t *r1, float_t *r2)
{
  Kuiper_Polymorphism1_kswap__float_t(r1, r2);
}

void Kuiper_Polymorphism1_swap_U64(uint64_t *r1, uint64_t *r2)
{
  uint64_t *gr1 = (uint64_t *)KPR_GPU_ALLOC((size_t)8U);
  uint64_t *gr2 = (uint64_t *)KPR_GPU_ALLOC((size_t)8U);
  MUST(cudaMemcpy(gr1, r1, (size_t)8U, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gr2, r2, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL_ASYNC(Kuiper_Polymorphism1_kswap__uint64_t, 1U, 1U, gr1, gr2);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(r1, gr1, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaMemcpy(r2, gr2, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr1));
  MUST(cudaFree(gr2));
}

void Kuiper_Polymorphism1_swap_F32(float_t *r1, float_t *r2)
{
  float_t *gr1 = (float_t *)KPR_GPU_ALLOC((size_t)4U);
  float_t *gr2 = (float_t *)KPR_GPU_ALLOC((size_t)4U);
  MUST(cudaMemcpy(gr1, r1, (size_t)4U, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gr2, r2, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL_ASYNC(Kuiper_Polymorphism1_kswap__float_t, 1U, 1U, gr1, gr2);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(r1, gr1, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaMemcpy(r2, gr2, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr1));
  MUST(cudaFree(gr2));
}

