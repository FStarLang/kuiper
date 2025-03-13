

#include "Kuiper_DotProduct3.h"

size_t Kuiper_DotProduct3_dp2_size = (size_t)1024U;

__device__

static void kernel(size_t nth, uint64_t *ga1, uint64_t *ga2, uint64_t *r)
{
  size_t tid = threadIdx_x();
  uint64_t vm = ga1[tid] * ga2[tid];
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  ar[tid] = vm;
  size_t tid1 = threadIdx_x();
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < nth)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = tid1 + (size_t)(1U << (uint32_t)it);
    if (nextid < nth)
      if ((tid1 & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        ar[tid1] += ar[nextid];
    n = it + (size_t)1U;
  }
  if (tid == (size_t)0U)
    *r = *ar;
}

__global__

static void __hoisted_0(uint64_t *ga1, uint64_t *ga2, uint64_t *gr)
{
  kernel((size_t)1024U, ga1, ga2, gr);
}

uint64_t Kuiper_DotProduct3_main(uint64_t *a1, uint64_t *a2)
{
  uint64_t *ar = (uint64_t *)KRML_HOST_CALLOC((size_t)1024U, sizeof (uint64_t));
  uint64_t *ga1 = (uint64_t *)KPR_GPU_ALLOC((size_t)8192U);
  uint64_t *ga2 = (uint64_t *)KPR_GPU_ALLOC((size_t)8192U);
  MUST(cudaMemcpy(ga1, a1, (size_t)8192U, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(ga2, a2, (size_t)8192U, cudaMemcpyHostToDevice));
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8192U);
  KPR_KCALL(__hoisted_0, (size_t)1U, (size_t)1024U, (size_t)8U, (size_t)1024U, ga1, ga2, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(ar, gr, (size_t)8192U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga1));
  MUST(cudaFree(ga2));
  MUST(cudaFree(gr));
  uint64_t dp = *ar;
  KRML_HOST_FREE(ar);
  return dp;
}

