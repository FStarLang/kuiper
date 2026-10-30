
#include "Kuiper_Example_FloatingNeg.h"

half Kuiper_Example_FloatingNeg_host_f16(half x) { return -x; }

__nv_bfloat16 Kuiper_Example_FloatingNeg_host_bf16(__nv_bfloat16 x)
{
    return -x;
}

float Kuiper_Example_FloatingNeg_host_f32(float x) { return -x; }

double Kuiper_Example_FloatingNeg_host_f64(double x) { return -x; }

half Kuiper_Example_FloatingNeg_host_double_neg_f16(half x) { return -(-x); }

__nv_bfloat16 Kuiper_Example_FloatingNeg_host_double_neg_bf16(__nv_bfloat16 x)
{
    return -(-x);
}

float Kuiper_Example_FloatingNeg_host_double_neg_f32(float x) { return -(-x); }

double Kuiper_Example_FloatingNeg_host_double_neg_f64(double x)
{
    return -(-x);
}

half Kuiper_Example_FloatingNeg_host_zero_sub_f16(half x)
{
    return __hsub(__float2half_rn(0.0f), x);
}

__nv_bfloat16 Kuiper_Example_FloatingNeg_host_zero_sub_bf16(__nv_bfloat16 x)
{
    return kpr_bf16sub(__float2bfloat16(0.0f), x);
}

float Kuiper_Example_FloatingNeg_host_zero_sub_f32(float x)
{
    return (float) 0LL - x;
}

double Kuiper_Example_FloatingNeg_host_zero_sub_f64(double x)
{
    return (double) 0LL - x;
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_f16
*/
static void
__hoisted_run_f16_0(uint32_t n, half *inputs, half *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -inputs[j];
    }
}

void Kuiper_Example_FloatingNeg_run_f16(uint32_t n, half *inputs, half *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_f16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_bf16
*/
static void
__hoisted_run_bf16_0(uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -inputs[j];
    }
}

void Kuiper_Example_FloatingNeg_run_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_bf16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_f32
*/
static void
__hoisted_run_f32_0(uint32_t n, float *inputs, float *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -inputs[j];
    }
}

void Kuiper_Example_FloatingNeg_run_f32(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_f32_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_f64
*/
static void
__hoisted_run_f64_0(uint32_t n, double *inputs, double *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -inputs[j];
    }
}

void Kuiper_Example_FloatingNeg_run_f64(
    uint32_t n, double *inputs, double *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_f64_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_double_neg_f16
*/
static void
__hoisted_run_double_neg_f16_0(uint32_t n, half *inputs, half *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -(-inputs[j]);
    }
}

void Kuiper_Example_FloatingNeg_run_double_neg_f16(
    uint32_t n, half *inputs, half *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_run_double_neg_f16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_double_neg_bf16
*/
static void
__hoisted_run_double_neg_bf16_0(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -(-inputs[j]);
    }
}

void Kuiper_Example_FloatingNeg_run_double_neg_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_run_double_neg_bf16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_double_neg_f32
*/
static void
__hoisted_run_double_neg_f32_0(uint32_t n, float *inputs, float *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -(-inputs[j]);
    }
}

void Kuiper_Example_FloatingNeg_run_double_neg_f32(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_run_double_neg_f32_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_double_neg_f64
*/
static void
__hoisted_run_double_neg_f64_0(uint32_t n, double *inputs, double *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = -(-inputs[j]);
    }
}

void Kuiper_Example_FloatingNeg_run_double_neg_f64(
    uint32_t n, double *inputs, double *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(
        __hoisted_run_double_neg_f64_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_zero_sub_f16
*/
static void
__hoisted_run_zero_sub_f16_0(uint32_t n, half *inputs, half *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = __hsub(__float2half_rn(0.0f), inputs[j]);
    }
}

void Kuiper_Example_FloatingNeg_run_zero_sub_f16(
    uint32_t n, half *inputs, half *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_zero_sub_f16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_zero_sub_bf16
*/
static void
__hoisted_run_zero_sub_bf16_0(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = kpr_bf16sub(__float2bfloat16(0.0f), inputs[j]);
    }
}

void Kuiper_Example_FloatingNeg_run_zero_sub_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_zero_sub_bf16_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_zero_sub_f32
*/
static void
__hoisted_run_zero_sub_f32_0(uint32_t n, float *inputs, float *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = (float) 0LL - inputs[j];
    }
}

void Kuiper_Example_FloatingNeg_run_zero_sub_f32(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_zero_sub_f32_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting run_zero_sub_f64
*/
static void
__hoisted_run_zero_sub_f64_0(uint32_t n, double *inputs, double *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        outputs[j] = (double) 0LL - inputs[j];
    }
}

void Kuiper_Example_FloatingNeg_run_zero_sub_f64(
    uint32_t n, double *inputs, double *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_run_zero_sub_f64_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
