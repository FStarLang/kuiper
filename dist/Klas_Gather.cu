
#include "Klas_Gather.h"

__global__
/**
  hoisted when extracting gather_bf16_u64_2d
*/
static void
__hoisted_gather_bf16_u64_2d_0(uint32_t cols,
                               uint32_t lenout,
                               __nv_bfloat16 *src,
                               uint64_t *idx, __nv_bfloat16 *out)
{
    if (1024U * blockIdx.x + threadIdx.x < lenout)
        out[1024U * blockIdx.x + threadIdx.x] =
            src[(uint32_t) idx[1024U * blockIdx.x + threadIdx.x] * cols +
                (1024U * blockIdx.x + threadIdx.x) % cols];
}

void
Klas_Gather_gather_bf16_u64_2d(uint32_t cols,
                               uint32_t lensrc,
                               uint32_t lenout,
                               __nv_bfloat16 *src,
                               uint64_t *idx, __nv_bfloat16 *out)
{
    KRML_MAYBE_UNUSED_VAR(lensrc);
    KPR_KCALL(__hoisted_gather_bf16_u64_2d_0,
              lenout / 1024U + (uint32_t) (lenout % 1024U != 0U),
              1024U, 0U, cols, lenout, src, idx, out);
    MUST(cudaDeviceSynchronize());
}
