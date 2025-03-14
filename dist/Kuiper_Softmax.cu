

#include "Kuiper_Softmax.h"

void Kuiper_Softmax_softmax_f16(size_t lena, half_t *a)
{
  half_t *ga = (half_t *)KPR_GPU_ALLOC((size_t)2U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)2U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(a, ga, (size_t)2U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

void Kuiper_Softmax_softmax_f32(size_t lena, float_t *a)
{
  float_t *ga = (float_t *)KPR_GPU_ALLOC((size_t)4U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)4U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(a, ga, (size_t)4U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

void Kuiper_Softmax_softmax_f64(size_t lena, double_t *a)
{
  double_t *ga = (double_t *)KPR_GPU_ALLOC((size_t)8U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)8U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(a, ga, (size_t)8U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

