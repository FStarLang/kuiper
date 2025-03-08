

#include "Kuiper_Softmax_F16.h"

__global__

static void k_reduce(size_t nth, half_t *a)
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
        a[tid] = __hadd(a[tid], a[nextid]);
    n = it + (size_t)1U;
  }
}

__global__

void Kuiper_Softmax_F16_k_pointwise_exp_f16(half_t *a)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t i = bid * bdim + threadIdx_x();
  a[i] = __hexp(a[i]);
}

__global__

void Kuiper_Softmax_F16_k_pointwise_div_f16(half_t *a, half_t d)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t i = bid * bdim + threadIdx_x();
  a[i] = __hdiv(a[i], d);
}

void Kuiper_Softmax_F16_softmax(size_t lena, half_t *a)
{
  half_t *ga = (half_t *)KPR_GPU_ALLOC((size_t)2U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)2U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL_ASYNC(Kuiper_Softmax_F16_k_pointwise_exp_f16, lena, 1U, ga);
  cudaDeviceSynchronize();
  half_t *a_ = (half_t *)KPR_GPU_ALLOC((size_t)2U * lena);
  MUST(cudaMemcpy(a_, ga, (size_t)2U * lena, cudaMemcpyDeviceToDevice));
  KPR_KCALL(k_reduce, (size_t)1U, lena, lena, a_);
  half_t *ca = (half_t *)KRML_HOST_MALLOC(sizeof (half_t));
  if (ca != NULL)
    *ca = (half_t)0.0f;
  MUST(cudaMemcpy(ca, a_, (size_t)2U, cudaMemcpyDeviceToHost));
  half_t x = *ca;
  KRML_HOST_FREE(ca);
  half_t avg = x;
  MUST(cudaFree(a_));
  KPR_KCALL_ASYNC(Kuiper_Softmax_F16_k_pointwise_div_f16, lena, 1U, ga, avg);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(a, ga, (size_t)2U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

