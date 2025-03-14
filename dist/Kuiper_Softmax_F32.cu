

#include "Kuiper_Softmax_F32.h"

void Kuiper_Softmax_F32_softmax(size_t lena, float_t *a)
{
  float_t *ga = (float_t *)KPR_GPU_ALLOC((size_t)4U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)4U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(a, ga, (size_t)4U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

