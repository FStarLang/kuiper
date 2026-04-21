
#include "Kuiper_Example_Polymorphism1.h"

__device__ static void kswap__uint64_t(uint64_t *r1, uint64_t *r2)
{
    uint64_t v11 = *r1;
    *r1 = *r2;
    *r2 = v11;
}

__global__
/**
  hoisted when extracting swap_U64
*/
static void __hoisted_0(uint64_t *gr1, uint64_t *gr2)
{
    kswap__uint64_t(gr1, gr2);
}

void Kuiper_Example_Polymorphism1_swap_U64(uint64_t *r1, uint64_t *r2)
{
    uint64_t *gr1 = (uint64_t *) KPR_GPU_ALLOC(8U, 1U);
    uint64_t *gr2 = (uint64_t *) KPR_GPU_ALLOC(8U, 1U);
    MUST(cudaMemcpy(gr1, r1, 8U, cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gr2, r2, 8U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U, gr1, gr2);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(r1, gr1, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(r2, gr2, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr1));
    MUST(cudaFree(gr2));
}

__device__ static void kswap__float(float *r1, float *r2)
{
    float v11 = *r1;
    *r1 = *r2;
    *r2 = v11;
}

__global__
/**
  hoisted when extracting swap_F32
*/
static void __hoisted_1(float *gr1, float *gr2)
{
    kswap__float(gr1, gr2);
}

void Kuiper_Example_Polymorphism1_swap_F32(float *r1, float *r2)
{
    float *gr1 = (float *)KPR_GPU_ALLOC(4U, 1U);
    float *gr2 = (float *)KPR_GPU_ALLOC(4U, 1U);
    MUST(cudaMemcpy(gr1, r1, 4U, cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gr2, r2, 4U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_1, 1U, 1U, 0U, gr1, gr2);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(r1, gr1, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(r2, gr2, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr1));
    MUST(cudaFree(gr2));
}
