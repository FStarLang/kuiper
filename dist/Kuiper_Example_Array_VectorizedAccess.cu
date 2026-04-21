
#include "Kuiper_Example_Array_VectorizedAccess.h"

__global__
/**
  hoisted when extracting hf
*/
static void __hoisted_0(float *a, float two)
{
    float local[4U];
    memset(local, 0U, 4U * sizeof(float));
    vec_memcpy(local, a);
    *local *= two;
    local[1U] *= two;
    local[2U] *= two;
    local[3U] *= two;
    vec_memcpy(a, local);
}

void Kuiper_Example_Array_VectorizedAccess_hf(float *v)
{
    float *a = (float *)KPR_GPU_ALLOC(4U, 4U);
    MUST(cudaMemcpy(a, v, 16U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U, a, 1.0f + 1.0f);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(v, a, 16U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(a));
}
