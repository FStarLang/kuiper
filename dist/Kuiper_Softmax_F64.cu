

#include "Kuiper_Softmax_F64.h"

void Kuiper_Softmax_F64_softmax(size_t lena, double_t *a)
{
  double_t *ga = (double_t *)KPR_GPU_ALLOC((size_t)8U * lena);
  MUST(cudaMemcpy(ga, a, (size_t)8U * lena, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(a, ga, (size_t)8U * lena, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
}

