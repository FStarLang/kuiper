
#include "Klas_Arange.h"

__global__
/**
  hoisted when extracting arange_i64
*/
static void __hoisted_arange_i64_0(uint32_t n, uint64_t start, uint64_t step,
                                   uint64_t *out)
{
    if (1024U * blockIdx.x + threadIdx.x < n)
        out[1024U * blockIdx.x + threadIdx.x] =
            start + step * (uint64_t) (uint32_t) (1024U * blockIdx.x +
                                                  threadIdx.x);
}

uint64_t *Klas_Arange_arange_i64(uint32_t n, uint64_t start, uint64_t step)
{
    uint64_t *out = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), n);
    KPR_KCALL(__hoisted_arange_i64_0,
              n / 1024U + (uint32_t) (n % 1024U != 0U),
              1024U, 0U, n, start, step, out);
    MUST(cudaDeviceSynchronize());
    return out;
}
