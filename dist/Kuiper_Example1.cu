
#include "Kuiper_Example1.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting main
*/
static void
__hoisted_main_0(uint64_t *gr)
{
    (*gr)++;
}

uint64_t Kuiper_Example1_main(void)
{
    uint64_t r = 1ULL;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC((uint32_t) sizeof(uint64_t), 1U);
    MUST(cudaMemcpy(gr, &r, (uint64_t) sizeof(uint64_t) * (uint64_t) 1U,
        cudaMemcpyHostToDevice));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_main_0, 1U, 1U, 0U, s, gr);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&r, gr, (uint64_t) sizeof(uint64_t) * (uint64_t) 1U,
        cudaMemcpyDeviceToHost));
    uint64_t v = r;
    MUST(cudaFree(gr));
    return v;
}

uint64_t Kuiper_Example1_test_device_to_device(void)
{
    uint64_t r = 2ULL;
    uint64_t *src = (uint64_t *) KPR_GPU_ALLOC((uint32_t) sizeof(uint64_t), 1U);
    uint64_t *dst = (uint64_t *) KPR_GPU_ALLOC((uint32_t) sizeof(uint64_t), 1U);
    MUST(cudaMemcpy(src, &r, (uint64_t) sizeof(uint64_t) * (uint64_t) 1U,
        cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(dst, src, (uint64_t) sizeof(uint64_t) * (uint64_t) 1U,
        cudaMemcpyDeviceToDevice));
    r = 0ULL;
    MUST(cudaMemcpy(&r, dst, (uint64_t) sizeof(uint64_t) * (uint64_t) 1U,
        cudaMemcpyDeviceToHost));
    uint64_t v = r;
    MUST(cudaFree(src));
    MUST(cudaFree(dst));
    return v;
}
