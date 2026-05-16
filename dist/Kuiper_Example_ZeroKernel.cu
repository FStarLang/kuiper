
#include "Kuiper_Example_ZeroKernel.h"

__global__
/**
  hoisted when extracting test
*/
static void __hoisted_test_0(void)
{

}

void Kuiper_Example_ZeroKernel_test(void)
{
    KPR_KCALL(__hoisted_test_0, 1U, 0U, 0U);
    MUST(cudaDeviceSynchronize());
}
