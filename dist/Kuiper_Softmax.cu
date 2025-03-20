

#include "Kuiper_Softmax.h"

__global__
static void __hoisted_0(float_t *ga)
{
  ga[blockIdx.x] = exp(ga[blockIdx_x]);
}

__global__
static void __hoisted_1(size_t lena, float_t *a_)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = threadIdx.x + (size_t)(1U << (uint32_t)it);
    if (nextid < lena)
      if ((threadIdx.x & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        a_[threadIdx.x] += a_[nextid];
    n = it + (size_t)1U;
  }
}

__global__
static void __hoisted_2(float_t *ga, float_t avg)
{
  ga[blockIdx.x] /= avg;
}

void Kuiper_Softmax_softmax_f32(size_t lena, float_t *a)
{
  float_t *ga = (float_t *)KPR_GPU_ALLOC((size_t)4U, lena);
  MUST(cudaMemcpy(ga, a, (size_t)4U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, lena, (size_t)1U, (size_t)1U, (size_t)0U, ga);
  cudaDeviceSynchronize();
  float_t *a_ = (float_t *)KPR_GPU_ALLOC((size_t)4U, lena);
  MUST(cudaMemcpy(a_, ga, (size_t)4U * lena, cudaMemcpyDeviceToDevice));
  KPR_KCALL(__hoisted_1, (size_t)1U, lena, (size_t)1U, (size_t)0U, lena, a_);
  cudaDeviceSynchronize();
  float_t *ca = (float_t *)KRML_HOST_MALLOC(sizeof (float_t));
  if (ca != NULL)
    *ca = (float_t)0.0f;
  MUST(cudaMemcpy(ca, a_, (size_t)4U, cudaMemcpyDeviceToHost));
  float_t x = *ca;
  KRML_HOST_FREE(ca);
  float_t avg = x;
  MUST(cudaFree(a_));
  KPR_KCALL(__hoisted_2, lena, (size_t)1U, (size_t)1U, (size_t)0U, ga, avg);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(a, ga, (size_t)4U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

__global__
static void __hoisted_3(double_t *ga)
{
  ga[blockIdx.x] = exp(ga[blockIdx_x]);
}

__global__
static void __hoisted_4(size_t lena, double_t *a_)
{
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < lena)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = threadIdx.x + (size_t)(1U << (uint32_t)it);
    if (nextid < lena)
      if ((threadIdx.x & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        a_[threadIdx.x] += a_[nextid];
    n = it + (size_t)1U;
  }
}

__global__
static void __hoisted_5(double_t *ga, double_t avg)
{
  ga[blockIdx.x] /= avg;
}

void Kuiper_Softmax_softmax_f64(size_t lena, double_t *a)
{
  double_t *ga = (double_t *)KPR_GPU_ALLOC((size_t)8U, lena);
  MUST(cudaMemcpy(ga, a, (size_t)8U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_3, lena, (size_t)1U, (size_t)1U, (size_t)0U, ga);
  cudaDeviceSynchronize();
  double_t *a_ = (double_t *)KPR_GPU_ALLOC((size_t)8U, lena);
  MUST(cudaMemcpy(a_, ga, (size_t)8U * lena, cudaMemcpyDeviceToDevice));
  KPR_KCALL(__hoisted_4, (size_t)1U, lena, (size_t)1U, (size_t)0U, lena, a_);
  cudaDeviceSynchronize();
  double_t *ca = (double_t *)KRML_HOST_MALLOC(sizeof (double_t));
  if (ca != NULL)
    *ca = (double_t)0.0l;
  MUST(cudaMemcpy(ca, a_, (size_t)8U, cudaMemcpyDeviceToHost));
  double_t x = *ca;
  KRML_HOST_FREE(ca);
  double_t avg = x;
  MUST(cudaFree(a_));
  KPR_KCALL(__hoisted_5, lena, (size_t)1U, (size_t)1U, (size_t)0U, ga, avg);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(a, ga, (size_t)8U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

