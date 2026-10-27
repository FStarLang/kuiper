
#include "Kuiper_Example_BFloat16Math.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run
*/
static void
__hoisted_run_0(uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        __nv_bfloat16 base = inputs[3U * j];
        __nv_bfloat16 coordinate = inputs[3U * j + 1U];
        __nv_bfloat16 position = inputs[3U * j + 2U];
        __nv_bfloat16 power =
            kpr_bf16pow(base, kpr_bf16div(coordinate, __ll2bfloat16_rn(64LL)));
        __nv_bfloat16 angle = kpr_bf16mul(position,
            kpr_bf16div(__float2bfloat16(1.0f),
                kpr_bf16pow(
                    base, kpr_bf16div(coordinate, __ll2bfloat16_rn(64LL)))));
        __nv_bfloat16 sn = kpr_bf16sin(angle);
        __nv_bfloat16 cs = kpr_bf16cos(angle);
        half bsn = __float2half_rn(__bfloat162float(sn));
        half bcs = __float2half_rn(__bfloat162float(cs));
        outputs[8U * j] = power;
        outputs[8U * j + 1U] = kpr_bf16sin(position);
        outputs[8U * j + 2U] = kpr_bf16cos(position);
        outputs[8U * j + 3U] = angle;
        outputs[8U * j + 4U] = sn;
        outputs[8U * j + 5U] = cs;
        outputs[8U * j + 6U] = __float2bfloat16(__half2float(bsn));
        outputs[8U * j + 7U] = __float2bfloat16(__half2float(bcs));
    }
}

void Kuiper_Example_BFloat16Math_run(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
