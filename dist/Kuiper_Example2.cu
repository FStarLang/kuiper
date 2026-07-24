
#include "Kuiper_Example2.h"

__global__
/**
  hoisted when extracting main
*/
static void __hoisted_main_0(void)
{

}

uint64_t Kuiper_Example2_main(void)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_main_0, 1U, 1U, 0U, s);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return 1ULL;
}
