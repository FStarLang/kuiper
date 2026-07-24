
#include "Kuiper_Example_Array_VectorizedAccess.h"

__global__
/**
  hoisted when extracting hf
*/
static void __hoisted_hf_0(float *a, float two)
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
    float *a = (float *)KPR_GPU_ALLOC(sizeof(float), 4U);
    MUST(cudaMemcpy
         (a, v, (uint32_t) sizeof(float) * 4U, cudaMemcpyHostToDevice));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_hf_0, 1U, 1U, 0U, s1, a, 1.0f + 1.0f);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy
         (v, a, (uint32_t) sizeof(float) * 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(a));
}
