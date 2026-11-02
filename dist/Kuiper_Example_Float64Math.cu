
#include "Kuiper_Example_Float64Math.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run
*/
static void
__hoisted_run_0(uint32_t n, double *inputs, double *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        double base = inputs[3U * j];
        double coordinate = inputs[3U * j + 1U];
        double position = inputs[3U * j + 2U];
        double power = pow(base, coordinate / (double) 64LL);
        double angle =
            position * ((double) 1LL / pow(base, coordinate / (double) 64LL));
        double sn = sin(angle);
        double cs = cos(angle);
        __nv_bfloat16 bsn = __float2bfloat16((float) sn);
        __nv_bfloat16 bcs = __float2bfloat16((float) cs);
        outputs[8U * j] = power;
        outputs[8U * j + 1U] = sin(position);
        outputs[8U * j + 2U] = cos(position);
        outputs[8U * j + 3U] = angle;
        outputs[8U * j + 4U] = sn;
        outputs[8U * j + 5U] = cs;
        outputs[8U * j + 6U] = (double) __bfloat162float(bsn);
        outputs[8U * j + 7U] = (double) __bfloat162float(bcs);
    }
}

void Kuiper_Example_Float64Math_run(uint32_t n, double *inputs, double *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
