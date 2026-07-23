
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_test_0, 1U, 0U, 0U, s);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
