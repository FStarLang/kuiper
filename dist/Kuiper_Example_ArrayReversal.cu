
#include "Kuiper_Example_ArrayReversal.h"

__global__
/**
  hoisted when extracting reverse_u64
*/
static void __hoisted_reverse_u64_0(uint32_t size, uint64_t *a)
{
    uint32_t idx_ = size - blockIdx.x - 1U;
    uint64_t uu = a[blockIdx.x];
    a[blockIdx.x] = a[idx_];
    a[idx_] = uu;
}

void Kuiper_Example_ArrayReversal_reverse_u64(uint32_t size, uint64_t *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_reverse_u64_0, size / 2U, 1U, 0U, s1, size, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
