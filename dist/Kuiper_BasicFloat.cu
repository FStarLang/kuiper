
#include "Kuiper_BasicFloat.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(float *gr)
{
    *gr += 1.0f;
}

float Kuiper_BasicFloat_main(void)
{
    float r = 0.0f;
    float *gr = (float *)KPR_GPU_ALLOC(4U, 1U);
    MUST(cudaMemcpy(gr, &r, 4U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, 4U, cudaMemcpyDeviceToHost));
    float v = r;
    MUST(cudaFree(gr));
    return v;
}
