

#include "Kuiper_Async1.h"

__global__

void Kuiper_Async1_kernel(uint64_t *r)
{
  *r += 1ULL;
}

uint64_t *Kuiper_Async1_galloc(uint64_t x)
{
  uint64_t r = x;
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  return gr;
}

uint64_t Kuiper_Async1_gread(uint64_t *gr)
{
  uint64_t r = 0ULL;
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
  return r;
}

uint64_t Kuiper_Async1_main(void)
{
  uint64_t *r1 = Kuiper_Async1_galloc(1ULL);
  uint64_t *r2 = Kuiper_Async1_galloc(2ULL);
  uint64_t *r3 = Kuiper_Async1_galloc(3ULL);
  uint64_t *r4 = Kuiper_Async1_galloc(4ULL);
  uint64_t *r5 = Kuiper_Async1_galloc(5ULL);
  uint64_t *r6 = Kuiper_Async1_galloc(6ULL);
  KPR_KCALL_ASYNC(Kuiper_Async1_kernel, 1U, 1U, r1);
  KPR_KCALL_ASYNC(Kuiper_Async1_kernel, 1U, 1U, r2);
  KPR_KCALL_ASYNC(Kuiper_Async1_kernel, 1U, 1U, r3);
  KPR_KCALL_ASYNC(Kuiper_Async1_kernel, 1U, 1U, r4);
  KPR_KCALL_ASYNC(Kuiper_Async1_kernel, 1U, 1U, r5);
  KPR_KCALL_ASYNC(Kuiper_Async1_kernel, 1U, 1U, r6);
  cudaDeviceSynchronize();
  uint64_t v1 = Kuiper_Async1_gread(r1);
  MUST(cudaFree(r1));
  uint64_t v2 = Kuiper_Async1_gread(r2);
  MUST(cudaFree(r2));
  uint64_t v3 = Kuiper_Async1_gread(r3);
  MUST(cudaFree(r3));
  uint64_t v4 = Kuiper_Async1_gread(r4);
  MUST(cudaFree(r4));
  uint64_t v5 = Kuiper_Async1_gread(r5);
  MUST(cudaFree(r5));
  uint64_t v6 = Kuiper_Async1_gread(r6);
  MUST(cudaFree(r6));
  return v1 + v2 + v3 + v4 + v5 + v6;
}

