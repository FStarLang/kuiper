

#include "Kuiper_Softmax_F64.h"

__global__

static void k_reduce(size_t nth, double_t *a)
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

__global__

void Kuiper_Softmax_F64_k_pointwise_exp_f64(double_t *a)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t i = bid * bdim + threadIdx_x();
  a[i] = exp(a[i]);
}

__global__

void Kuiper_Softmax_F64_k_pointwise_div_f64(double_t *a, double_t d)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t i = bid * bdim + threadIdx_x();
  a[i] /= d;
}

void Kuiper_Softmax_F64_softmax(size_t lena, double_t *a)
{
  double_t *ga = (double_t *)KPR_GPU_ALLOC((size_t)8U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)8U * lena, cudaMemcpyHostToDevice));
  KPR_KCALL_ASYNC(Kuiper_Softmax_F64_k_pointwise_exp_f64, lena, 1U, ga);
  cudaDeviceSynchronize();
  double_t *a_ = (double_t *)KPR_GPU_ALLOC((size_t)8U * lena);
  MUST(cudaMemcpy(a_, ga, (size_t)8U * lena, cudaMemcpyDeviceToDevice));
  KPR_KCALL_SHMEM_ASYNC(k_reduce, (size_t)1U, lena, (size_t)4U, (size_t)0U, lena, a_);
  cudaDeviceSynchronize();
  double_t *ca = (double_t *)KRML_HOST_MALLOC(sizeof (double_t));
  if (ca != NULL)
    *ca = (double_t)0.0l;
  MUST(cudaMemcpy(ca, a_, (size_t)8U, cudaMemcpyDeviceToHost));
  double_t x = *ca;
  KRML_HOST_FREE(ca);
  double_t avg = x;
  MUST(cudaFree(a_));
  KPR_KCALL_ASYNC(Kuiper_Softmax_F64_k_pointwise_div_f64, lena, 1U, ga, avg);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(a, ga, (size_t)8U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

