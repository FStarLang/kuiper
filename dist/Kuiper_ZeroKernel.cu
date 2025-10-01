

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
  KPR_KCALL(__hoisted_0, (uint32_t)0U, (uint32_t)0U, (uint32_t)0U);
  cudaDeviceSynchronize();
}

