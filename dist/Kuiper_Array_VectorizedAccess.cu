

#include "Kuiper_Array_VectorizedAccess.h"

__global__
/**
  hoisted when extracting hf
*/
static void __hoisted_0(float_t *a, float_t two)
{
  float4 fv = KPR_VECTZD_READ(a, (size_t)0U);
  float_t x = KPR_PROJ_X(fv) * two;
  float_t y = KPR_PROJ_Y(fv) * two;
  float_t z = KPR_PROJ_Z(fv) * two;
  KPR_VECTZD_WRITE(a, (size_t)0U, make_float4(x, y, z, KPR_PROJ_W(fv) * two));
}

void Kuiper_Array_VectorizedAccess_hf(float_t *v)
{
  float_t *a = (float_t *)KPR_GPU_ALLOC((size_t)4U, (size_t)4U);
  MUST(cudaMemcpy(a, v, (size_t)16U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, (size_t)1U, (size_t)1U, (size_t)0U, a, (float_t)1.0f + (float_t)1.0f);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(v, a, (size_t)16U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(a));
}

