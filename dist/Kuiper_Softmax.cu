

#include "Kuiper_Softmax.h"

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_0(float_t *ga)
{
  ga[blockIdx.x] = exp(ga[blockIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_1(uint32_t lena, float_t *a_)
{
  uint32_t n = 0U;
  for (; (uint32_t)(1U << (uint32_t)n) < lena; n += 1U)
  {
    uint32_t __anf0 = n;
    __syncthreads();
    uint32_t nextid = threadIdx.x + (uint32_t)(1U << (uint32_t)__anf0);
    if (nextid < lena)
      if ((threadIdx.x & (uint32_t)(1U << (uint32_t)(__anf0 + 1U)) - 1U) == 0U)
        a_[threadIdx.x] += a_[nextid];
  }
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_2(float_t *ga, float_t avg)
{
  ga[blockIdx.x] /= avg;
}

void Kuiper_Softmax_softmax_f32(uint32_t lena, float_t *a)
{
  float_t *ga = (float_t *)KPR_GPU_ALLOC(4U, lena);
  MUST(cudaMemcpy(ga, a, 4U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, lena, 1U, 0U, ga);
  cudaDeviceSynchronize();
  float_t *a_ = (float_t *)KPR_GPU_ALLOC(4U, lena);
  MUST(cudaMemcpy(a_, ga, 4U * lena, cudaMemcpyDeviceToDevice));
  KPR_KCALL(__hoisted_1, 1U, lena, 0U, lena, a_);
  cudaDeviceSynchronize();
  float_t *ca = (float_t *)KRML_HOST_MALLOC(sizeof (float_t));
  if (ca != NULL)
    *ca = 0.0f;
  MUST(cudaMemcpy(ca, a_, 4U, cudaMemcpyDeviceToHost));
  float_t x = *ca;
  KRML_HOST_FREE(ca);
  float_t avg = x;
  MUST(cudaFree(a_));
  KPR_KCALL(__hoisted_2, lena, 1U, 0U, ga, avg);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(a, ga, 4U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_3(double_t *ga)
{
  ga[blockIdx.x] = exp(ga[blockIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_4(uint32_t lena, double_t *a_)
{
  uint32_t n = 0U;
  for (; (uint32_t)(1U << (uint32_t)n) < lena; n += 1U)
  {
    uint32_t __anf0 = n;
    __syncthreads();
    uint32_t nextid = threadIdx.x + (uint32_t)(1U << (uint32_t)__anf0);
    if (nextid < lena)
      if ((threadIdx.x & (uint32_t)(1U << (uint32_t)(__anf0 + 1U)) - 1U) == 0U)
        a_[threadIdx.x] += a_[nextid];
  }
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_5(double_t *ga, double_t avg)
{
  ga[blockIdx.x] /= avg;
}

void Kuiper_Softmax_softmax_f64(uint32_t lena, double_t *a)
{
  double_t *ga = (double_t *)KPR_GPU_ALLOC(8U, lena);
  MUST(cudaMemcpy(ga, a, 8U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_3, lena, 1U, 0U, ga);
  cudaDeviceSynchronize();
  double_t *a_ = (double_t *)KPR_GPU_ALLOC(8U, lena);
  MUST(cudaMemcpy(a_, ga, 8U * lena, cudaMemcpyDeviceToDevice));
  KPR_KCALL(__hoisted_4, 1U, lena, 0U, lena, a_);
  cudaDeviceSynchronize();
  double_t *ca = (double_t *)KRML_HOST_MALLOC(sizeof (double_t));
  if (ca != NULL)
    *ca = 0.0l;
  MUST(cudaMemcpy(ca, a_, 8U, cudaMemcpyDeviceToHost));
  double_t x = *ca;
  KRML_HOST_FREE(ca);
  double_t avg = x;
  MUST(cudaFree(a_));
  KPR_KCALL(__hoisted_5, lena, 1U, 0U, ga, avg);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(a, ga, 8U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

