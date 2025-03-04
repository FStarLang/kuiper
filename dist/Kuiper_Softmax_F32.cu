

#include "Kuiper_Softmax_F32.h"

__global__

static void k_reduce(size_t nth, float_t *a)
{
  size_t tid = threadIdx_x();
  size_t n = (size_t)0U;
  while ((size_t)(1U << (uint32_t)n) < nth)
  {
    size_t it = n;
    __syncthreads();
    size_t nextid = tid + (size_t)(1U << (uint32_t)it);
    if (nextid < nth)
      if ((tid & (size_t)(1U << (uint32_t)(it + (size_t)1U)) - (size_t)1U) == (size_t)0U)
        a[tid] += a[nextid];
    n = it + (size_t)1U;
  }
}

static void reduce(size_t lena, float_t *a)
{
  KPR_KCALL(k_reduce, (size_t)1U, lena, lena, a);
}

__global__

void Kuiper_Softmax_F32_k_pointwise_exp(float_t *a)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t i = bid * bdim + threadIdx_x();
  a[i] = exp(a[i]);
}

__global__

void Kuiper_Softmax_F32_k_pointwise_div(float_t *a, float_t d)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t i = bid * bdim + threadIdx_x();
  a[i] /= d;
}

float_t Kuiper_Softmax_F32_arr_read_1(float_t *a)
{
  float_t *ca = (float_t *)KRML_HOST_MALLOC(sizeof (float_t));
  if (ca != NULL)
    *ca = (float_t)0.0f;
  MUST(cudaMemcpy(ca, a, (size_t)4U, cudaMemcpyDeviceToHost));
  float_t x = *ca;
  KRML_HOST_FREE(ca);
  return x;
}

void Kuiper_Softmax_F32_softmax_gpu(size_t lena, float_t *a)
{
  KPR_KCALL(Kuiper_Softmax_F32_k_pointwise_exp, lena, 1U, a);
  float_t *a_ = (float_t *)KPR_GPU_ALLOC((size_t)4U * lena);
  MUST(cudaMemcpy(a_, a, (size_t)4U * lena, cudaMemcpyDeviceToDevice));
  reduce(lena, a_);
  float_t avg = Kuiper_Softmax_F32_arr_read_1(a_);
  MUST(cudaFree(a_));
  KPR_KCALL(Kuiper_Softmax_F32_k_pointwise_div, lena, 1U, a, avg);
}

void Kuiper_Softmax_F32_softmax(size_t lena, float_t *a)
{
  float_t *ga = (float_t *)KPR_GPU_ALLOC((size_t)4U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)4U * lena, cudaMemcpyHostToDevice));
  Kuiper_Softmax_F32_softmax_gpu(lena, ga);
  MUST(cudaMemcpy(a, ga, (size_t)4U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

