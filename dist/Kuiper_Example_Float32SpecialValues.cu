
#include "Kuiper_Example_Float32SpecialValues.h"

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
        float shifted = __fsub_rn(-INFINITY, x);
        float padded = expf(__fsub_rn(-INFINITY, x));
        float difference = __fsub_rn(x, (float) 1LL);
        outputs[4U * j] = expf(x);
        outputs[4U * j + 1U] = shifted;
        outputs[4U * j + 2U] = padded;
        outputs[4U * j + 3U] = difference;
    }
}

void Kuiper_Example_Float32SpecialValues_run(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_exp_f16
*/
static void
__hoisted_run_exp_f16_0(uint32_t n, half *inputs, half *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = hexp(inputs[j]);
    }
}

void Kuiper_Example_Float32SpecialValues_run_exp_f16(
    uint32_t n, half *inputs, half *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_exp_f16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_exp_bf16
*/
static void
__hoisted_run_exp_bf16_0(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = kpr_bf16exp(inputs[j]);
    }
}

void Kuiper_Example_Float32SpecialValues_run_exp_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_exp_bf16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_exp_f64
*/
static void
__hoisted_run_exp_f64_0(uint32_t n, double *inputs, double *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = exp(inputs[j]);
    }
}

void Kuiper_Example_Float32SpecialValues_run_exp_f64(
    uint32_t n, double *inputs, double *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_exp_f64_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_subtraction
*/
static void
__hoisted_run_subtraction_0(
    uint32_t n, float *left, float *right, float *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        float x = left[j];
        float y = right[j];
        float first = __fsub_rn(x, y);
        __fsub_rn(x, y);
        float first0 = first;
        float second = __fsub_rn(x, y);
        outputs[2U * j] = first0;
        outputs[2U * j + 1U] = second;
    }
}

void Kuiper_Example_Float32SpecialValues_run_subtraction(
    uint32_t n, float *left, float *right, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_run_subtraction_0, 1U, 1U, 0U, s, n, left, right, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
