
#include "Klas_Misc.h"

__global__
/**
  hoisted when extracting arange_i64
*/
static void __hoisted_arange_i64_0(uint32_t len, uint64_t *out)
{
    if (1024U * blockIdx.x + threadIdx.x < len)
        out[1024U * blockIdx.x + threadIdx.x] =
            FStar_UInt64_uint_to_t(FStar_SizeT_v
                                   (1024U * blockIdx.x + threadIdx.x));
}

void Klas_Misc_arange_i64(uint32_t len, uint64_t *out)
{
    KPR_KCALL(__hoisted_arange_i64_0,
              len / 1024U + (uint32_t) (len % 1024U != 0U),
              1024U, 0U, len, out);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting gather_bf16_u32
*/
static void
__hoisted_gather_bf16_u32_0(uint32_t lens,
                            uint32_t leni,
                            __nv_bfloat16 *src,
                            uint32_t *idx, __nv_bfloat16 *out)
{
    if (1024U * blockIdx.x + threadIdx.x < leni)
        out[1024U * blockIdx.x + threadIdx.x] =
            src[(uint32_t) idx[1024U * blockIdx.x + threadIdx.x] % lens];
}

void
Klas_Misc_gather_bf16_u32(uint32_t lens,
                          uint32_t leni,
                          __nv_bfloat16 *src, uint32_t *idx, __nv_bfloat16 *out)
{
    KPR_KCALL(__hoisted_gather_bf16_u32_0,
              leni / 1024U + (uint32_t) (leni % 1024U != 0U),
              1024U, 0U, lens, leni, src, idx, out);
    MUST(cudaDeviceSynchronize());
}
