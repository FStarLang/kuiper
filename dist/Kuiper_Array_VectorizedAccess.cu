

#include "Kuiper_Array_VectorizedAccess.h"

__global__
/**
  hoisted when extracting hf
*/
static void __hoisted_0(float_t *a, float_t two)
{
  float_t local[4U];
  memset(local, 0U, (uint32_t)4U * sizeof (float_t));
  vec_memcpy(local, a);
  *local *= two;
  local[1U] *= two;
  local[2U] *= two;
  local[3U] *= two;
  vec_memcpy(a, local);
}

void Kuiper_Array_VectorizedAccess_hf(float_t *v)
{
  float_t *a = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, (uint32_t)4U);
  MUST(cudaMemcpy(a, v, (uint32_t)16U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0,
    (uint32_t)1U,
    (uint32_t)1U,
    (uint32_t)0U,
    a,
    (float_t)1.0f + (float_t)1.0f);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(v, a, (uint32_t)16U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(a));
}

