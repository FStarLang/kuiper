

#include "Kuiper_DotProduct.h"

__global__

static void __hoisted_0(size_t lena, float_t *ga1, float_t *ga2)
{
  ga1[threadIdx.x] *= ga2[threadIdx_x];
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = threadIdx.x + (size_t)(1U << (uint32_t)it);
    if (nextid < lena)
      if ((threadIdx.x & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        ga1[threadIdx.x] += ga1[nextid];
    n = it + (size_t)1U;
  }
}

float_t Kuiper_DotProduct_dotprod_f32(size_t lena, float_t *a1, float_t *a2)
{
  float_t *ar = (float_t *)KRML_HOST_MALLOC(sizeof (float_t));
  if (ar != NULL)
    *ar = (float_t)0.0f;
  float_t *ga1 = (float_t *)KPR_GPU_ALLOC((size_t)4U, lena);
  float_t *ga2 = (float_t *)KPR_GPU_ALLOC((size_t)4U, lena);
  MUST(cudaMemcpy(ga1, a1, (size_t)4U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(ga2, a2, (size_t)4U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, (size_t)1U, lena, (size_t)1U, (size_t)0U, lena, ga1, ga2);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(ar, ga1, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga1));
  MUST(cudaFree(ga2));
  float_t dp = *ar;
  KRML_HOST_FREE(ar);
  return dp;
}

__global__

static void __hoisted_1(size_t lena, double_t *ga1, double_t *ga2)
{
  ga1[threadIdx.x] *= ga2[threadIdx_x];
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = threadIdx.x + (size_t)(1U << (uint32_t)it);
    if (nextid < lena)
      if ((threadIdx.x & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        ga1[threadIdx.x] += ga1[nextid];
    n = it + (size_t)1U;
  }
}

double_t Kuiper_DotProduct_dotprod_f64(size_t lena, double_t *a1, double_t *a2)
{
  double_t *ar = (double_t *)KRML_HOST_MALLOC(sizeof (double_t));
  if (ar != NULL)
    *ar = (double_t)0.0l;
  double_t *ga1 = (double_t *)KPR_GPU_ALLOC((size_t)8U, lena);
  double_t *ga2 = (double_t *)KPR_GPU_ALLOC((size_t)8U, lena);
  MUST(cudaMemcpy(ga1, a1, (size_t)8U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(ga2, a2, (size_t)8U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_1, (size_t)1U, lena, (size_t)1U, (size_t)0U, lena, ga1, ga2);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(ar, ga1, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga1));
  MUST(cudaFree(ga2));
  double_t dp = *ar;
  KRML_HOST_FREE(ar);
  return dp;
}

__global__

static void __hoisted_2(size_t lena, uint32_t *ga1, uint32_t *ga2)
{
  ga1[threadIdx.x] *= ga2[threadIdx_x];
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = threadIdx.x + (size_t)(1U << (uint32_t)it);
    if (nextid < lena)
      if ((threadIdx.x & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        ga1[threadIdx.x] += ga1[nextid];
    n = it + (size_t)1U;
  }
}

uint32_t Kuiper_DotProduct_dotprod_u32(size_t lena, uint32_t *a1, uint32_t *a2)
{
  uint32_t *ar = (uint32_t *)KRML_HOST_CALLOC((size_t)1U, sizeof (uint32_t));
  uint32_t *ga1 = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, lena);
  uint32_t *ga2 = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, lena);
  MUST(cudaMemcpy(ga1, a1, (size_t)4U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(ga2, a2, (size_t)4U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_2, (size_t)1U, lena, (size_t)1U, (size_t)0U, lena, ga1, ga2);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(ar, ga1, (size_t)4U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga1));
  MUST(cudaFree(ga2));
  uint32_t dp = *ar;
  KRML_HOST_FREE(ar);
  return dp;
}

__global__

static void __hoisted_3(size_t lena, uint64_t *ga1, uint64_t *ga2)
{
  ga1[threadIdx.x] *= ga2[threadIdx_x];
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = threadIdx.x + (size_t)(1U << (uint32_t)it);
    if (nextid < lena)
      if ((threadIdx.x & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        ga1[threadIdx.x] += ga1[nextid];
    n = it + (size_t)1U;
  }
}

uint64_t Kuiper_DotProduct_dotprod_u64(size_t lena, uint64_t *a1, uint64_t *a2)
{
  uint64_t *ar = (uint64_t *)KRML_HOST_CALLOC((size_t)1U, sizeof (uint64_t));
  uint64_t *ga1 = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, lena);
  uint64_t *ga2 = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, lena);
  MUST(cudaMemcpy(ga1, a1, (size_t)8U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(ga2, a2, (size_t)8U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_3, (size_t)1U, lena, (size_t)1U, (size_t)0U, lena, ga1, ga2);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(ar, ga1, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga1));
  MUST(cudaFree(ga2));
  uint64_t dp = *ar;
  KRML_HOST_FREE(ar);
  return dp;
}

