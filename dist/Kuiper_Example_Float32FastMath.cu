
#include "Kuiper_Example_Float32FastMath.h"

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
        float x = inputs[3U * j];
        float y = inputs[3U * j + 1U];
        float z = inputs[3U * j + 2U];
        float e = __expf(x);
        float d = __fdividef(x, y);
        float f = __fmaf_rn(x, y, z);
        float s1 = __fsub_rn(x, y);
        outputs[4U * j] = e;
        outputs[4U * j + 1U] = d;
        outputs[4U * j + 2U] = f;
        outputs[4U * j + 3U] = s1;
    }
}

void Kuiper_Example_Float32FastMath_run(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
