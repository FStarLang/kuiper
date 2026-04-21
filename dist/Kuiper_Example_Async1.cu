
#include "Kuiper_Example_Async1.h"

uint64_t *Kuiper_Example_Async1_galloc(uint64_t x)
{
    uint64_t r = x;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(8U, 1U);
    MUST(cudaMemcpy(gr, &r, 8U, cudaMemcpyHostToDevice));
    return gr;
}

uint64_t Kuiper_Example_Async1_gread(uint64_t *gr)
{
    uint64_t r = 0ULL;
    MUST(cudaMemcpy(&r, gr, 8U, cudaMemcpyDeviceToHost));
    return r;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(uint64_t *r1)
{
    *r1 += 1ULL;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_1(uint64_t *r2)
{
    *r2 += 1ULL;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_2(uint64_t *r3)
{
    *r3 += 1ULL;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_3(uint64_t *r4)
{
    *r4 += 1ULL;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_4(uint64_t *r5)
{
    *r5 += 1ULL;
}

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_5(uint64_t *r6)
{
    *r6 += 1ULL;
}

uint64_t Kuiper_Example_Async1_main(void)
{
    uint64_t *r1 = Kuiper_Example_Async1_galloc(1ULL);
    uint64_t *r2 = Kuiper_Example_Async1_galloc(2ULL);
    uint64_t *r3 = Kuiper_Example_Async1_galloc(3ULL);
    uint64_t *r4 = Kuiper_Example_Async1_galloc(4ULL);
    uint64_t *r5 = Kuiper_Example_Async1_galloc(5ULL);
    uint64_t *r6 = Kuiper_Example_Async1_galloc(6ULL);
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U, r1);
    KPR_KCALL(__hoisted_1, 1U, 1U, 0U, r2);
    KPR_KCALL(__hoisted_2, 1U, 1U, 0U, r3);
    KPR_KCALL(__hoisted_3, 1U, 1U, 0U, r4);
    KPR_KCALL(__hoisted_4, 1U, 1U, 0U, r5);
    KPR_KCALL(__hoisted_5, 1U, 1U, 0U, r6);
    MUST(cudaDeviceSynchronize());
    uint64_t v1 = Kuiper_Example_Async1_gread(r1);
    MUST(cudaFree(r1));
    uint64_t v2 = Kuiper_Example_Async1_gread(r2);
    MUST(cudaFree(r2));
    uint64_t v3 = Kuiper_Example_Async1_gread(r3);
    MUST(cudaFree(r3));
    uint64_t v4 = Kuiper_Example_Async1_gread(r4);
    MUST(cudaFree(r4));
    uint64_t v5 = Kuiper_Example_Async1_gread(r5);
    MUST(cudaFree(r5));
    uint64_t v6 = Kuiper_Example_Async1_gread(r6);
    MUST(cudaFree(r6));
    return v1 + v2 + v3 + v4 + v5 + v6;
}
