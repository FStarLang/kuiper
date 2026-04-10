
#include "Kuiper_ZeroKernel.h"

__global__
/**
  hoisted when extracting test
*/
static void __hoisted_0(void)
{

}

void Kuiper_ZeroKernel_test(void)
{
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, 1U, 0U, 0U);
    MUST(cudaDeviceSynchronize());
}
