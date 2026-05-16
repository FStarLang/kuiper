
#include "Kuiper_Example1.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(uint64_t *gr)
{
    (*gr)++;
}

uint64_t Kuiper_Example1_main(void)
{
    uint64_t r = 1ULL;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(uint64_t), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t v = r;
    MUST(cudaFree(gr));
    return v;
}
