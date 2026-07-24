
#include "Kuiper_Example1.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_0(uint64_t *gr)
{
    (*gr)++;
}

uint64_t Kuiper_Example1_main(void)
{
    uint64_t r = 1ULL;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(uint64_t), cudaMemcpyHostToDevice));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_main_0, 1U, 1U, 0U, s, gr);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&r, gr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t v = r;
    MUST(cudaFree(gr));
    return v;
}
