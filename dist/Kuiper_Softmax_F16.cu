

#include "Kuiper_Softmax_F16.h"

void Kuiper_Softmax_F16_softmax(size_t lena, half_t *a)
{
  half_t *ga = (half_t *)KPR_GPU_ALLOC((size_t)2U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)2U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(a, ga, (size_t)2U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

