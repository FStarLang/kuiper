
#include "Kuiper_Example_Float32Math.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run
*/
static void
__hoisted_run_0(uint32_t n, float *inputs, float *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        float base = inputs[3U * j];
        float coordinate = inputs[3U * j + 1U];
        float position = inputs[3U * j + 2U];
        float power = powf(base, coordinate / (float) 64LL);
        float angle =
            position * ((float) 1LL / powf(base, coordinate / (float) 64LL));
        float sn = sinf(angle);
        float cs = cosf(angle);
        __nv_bfloat16 bsn = __float2bfloat16(sn);
        __nv_bfloat16 bcs = __float2bfloat16(cs);
        outputs[8U * j] = power;
        outputs[8U * j + 1U] = sinf(position);
        outputs[8U * j + 2U] = cosf(position);
        outputs[8U * j + 3U] = angle;
        outputs[8U * j + 4U] = sn;
        outputs[8U * j + 5U] = cs;
        outputs[8U * j + 6U] = __bfloat162float(bsn);
        outputs[8U * j + 7U] = __bfloat162float(bcs);
    }
}

void Kuiper_Example_Float32Math_run(uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
