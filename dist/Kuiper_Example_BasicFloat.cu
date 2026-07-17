
#include "Kuiper_Example_BasicFloat.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_0(float *gr)
{
    *gr += 1.0f;
}

float Kuiper_Example_BasicFloat_main(void)
{
    float r = 0.0f;
    float *gr = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(float), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_main_0, 1U, 1U, 0U, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof(float), cudaMemcpyDeviceToHost));
    float v = r;
    MUST(cudaFree(gr));
    return v;
}
