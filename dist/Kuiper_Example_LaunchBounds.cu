
#include "Kuiper_Example_LaunchBounds.h"

__global__ __launch_bounds__(32)
/**
  hoisted when extracting fixed32
*/
static void
__hoisted_fixed32_0(void)
{
}

void Kuiper_Example_LaunchBounds_fixed32(void)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_fixed32_0, 1U, 32U, 0U, s);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(64)
/**
  hoisted when extracting fixed64
*/
static void
__hoisted_fixed64_0(void)
{
}

void Kuiper_Example_LaunchBounds_fixed64(void)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_fixed64_0, 1U, 64U, 0U, s);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting dynamic
*/
static void
__hoisted_dynamic_0(void)
{
}

void Kuiper_Example_LaunchBounds_dynamic(uint32_t nthr)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_dynamic_0, 1U, nthr, 0U, s);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting zero
*/
static void
__hoisted_zero_0(void)
{
}

void Kuiper_Example_LaunchBounds_zero(void)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_zero_0, 1U, 0U, 0U, s);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
