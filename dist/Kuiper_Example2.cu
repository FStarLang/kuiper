

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
  KPR_KCALL(__hoisted_0, (size_t)1U, (size_t)1U, (size_t)0U);
  cudaDeviceSynchronize();
  return 1ULL;
}

