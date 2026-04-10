
#include "Kuiper_Example2.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_0(void)
{

}

uint64_t Kuiper_Example2_main(void)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, 1U, 1U, 0U);
    MUST(cudaDeviceSynchronize());
    return 1ULL;
}
