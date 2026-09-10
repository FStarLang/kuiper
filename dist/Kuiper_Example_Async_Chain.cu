
#include "Kuiper_Example_Async_Chain.h"

uint64_t *Kuiper_Example_Async_Chain_galloc(uint64_t x)
{
    uint64_t r = x;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(uint64_t), cudaMemcpyHostToDevice));
    return gr;
}

uint64_t Kuiper_Example_Async_Chain_gread(uint64_t *gr)
{
    uint64_t r = 0ULL;
    MUST(cudaMemcpy(&r, gr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return r;
}

__global__
/**
  hoisted when extracting main
*/
static void
__hoisted_main_0(uint64_t *r)
{
    (*r)++;
}

__global__
/**
  hoisted when extracting main
*/
static void
__hoisted_main_1(uint64_t *r)
{
    (*r)++;
}

__global__
/**
  hoisted when extracting main
*/
static void
__hoisted_main_2(uint64_t *r)
{
    (*r)++;
}

uint64_t Kuiper_Example_Async_Chain_main(void)
{
    uint64_t *r = Kuiper_Example_Async_Chain_galloc(1ULL);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_main_0, 1U, 1U, 0U, s, r);
    KPR_KCALL(__hoisted_main_1, 1U, 1U, 0U, s, r);
    KPR_KCALL(__hoisted_main_2, 1U, 1U, 0U, s, r);
    MUST(cudaStreamSynchronize(s));
    uint64_t v = Kuiper_Example_Async_Chain_gread(r);
    MUST(cudaFree(r));
    MUST(cudaStreamDestroy(s));
    KPR_ASSERT(v == 4ULL);
    return v;
}
