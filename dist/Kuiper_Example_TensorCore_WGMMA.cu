
#include "Kuiper_Example_TensorCore_WGMMA.h"

inline __device__ void Kuiper_Example_TensorCore_WGMMA_accumulate_twice(
    __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    auto &fr = KPR_INIT(kpr_wgmma_fragment);
    kpr_wgmma_load_accum(fr, c, 8U);
    kpr_wgmma_mma_sync(a, b, fr);
    kpr_wgmma_mma_sync(a, b, fr);
    kpr_wgmma_store(fr, c, 8U);
}

inline __device__ void Kuiper_Example_TensorCore_WGMMA_multiply(
    __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    auto &fr = KPR_INIT(kpr_wgmma_fragment);
    kpr_wgmma_fill(fr, 0.0f);
    kpr_wgmma_mma_sync(a, b, fr);
    kpr_wgmma_store(fr, c, 8U);
}
