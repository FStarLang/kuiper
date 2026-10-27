
#include "Kuiper_Example_Float16Math.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run
*/
static void
__hoisted_run_0(uint32_t n, half *inputs, half *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        half base = inputs[3U * j];
        half coordinate = inputs[3U * j + 1U];
        half position = inputs[3U * j + 2U];
        half power = kpr_hpow(base, __hdiv(coordinate, __ll2half_rn(64LL)));
        half angle = __hmul(position,
            __hdiv(__float2half_rn(1.0f),
                kpr_hpow(base, __hdiv(coordinate, __ll2half_rn(64LL)))));
        half sn = kpr_hsin(angle);
        half cs = kpr_hcos(angle);
        __nv_bfloat16 bsn = __float2bfloat16(__half2float(sn));
        __nv_bfloat16 bcs = __float2bfloat16(__half2float(cs));
        outputs[8U * j] = power;
        outputs[8U * j + 1U] = kpr_hsin(position);
        outputs[8U * j + 2U] = kpr_hcos(position);
        outputs[8U * j + 3U] = angle;
        outputs[8U * j + 4U] = sn;
        outputs[8U * j + 5U] = cs;
        outputs[8U * j + 6U] = __float2half_rn(__bfloat162float(bsn));
        outputs[8U * j + 7U] = __float2half_rn(__bfloat162float(bcs));
    }
}

void Kuiper_Example_Float16Math_run(uint32_t n, half *inputs, half *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
