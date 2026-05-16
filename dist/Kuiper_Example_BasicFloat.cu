
#include "Kuiper_Example_BasicFloat.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(float *gr)
{
    *gr += 1.0f;
}

float Kuiper_Example_BasicFloat_main(void)
{
    float r = 0.0f;
    float *gr = (float *)KPR_GPU_ALLOC(sizeof((float) 0), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof((float) 0), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof((float) 0), cudaMemcpyDeviceToHost));
    float v = r;
    MUST(cudaFree(gr));
    return v;
}
