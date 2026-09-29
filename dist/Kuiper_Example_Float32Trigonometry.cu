
#include "Kuiper_Example_Float32Trigonometry.h"

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
        float x = inputs[j];
        float s = __sinf(x);
        float c = __cosf(x);
        outputs[2U * j] = s;
        outputs[2U * j + 1U] = c;
    }
}

void Kuiper_Example_Float32Trigonometry_run(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
