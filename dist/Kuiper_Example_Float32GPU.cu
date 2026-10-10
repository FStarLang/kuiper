
#include "Kuiper_Example_Float32GPU.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting multiply
*/
static void
__hoisted_multiply_0(float x, float y, float *device)
{
    *device = kpr_f32_mul_rn_ftz(x, y);
}

float Kuiper_Example_Float32GPU_multiply(float x, float y)
{
    float out = (float) 0LL;
    float *device = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_multiply_0, 1U, 1U, 0U, s, x, y, device);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&out, device, sizeof(float), cudaMemcpyDeviceToHost));
    float result = out;
    MUST(cudaFree(device));
    return result;
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting exponentiate
*/
static void
__hoisted_exponentiate_0(float x, float *device)
{
    *device = kpr_f32_exp2_approx_ftz(x);
}

float Kuiper_Example_Float32GPU_exponentiate(float x)
{
    float out = (float) 0LL;
    float *device = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_exponentiate_0, 1U, 1U, 0U, s, x, device);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&out, device, sizeof(float), cudaMemcpyDeviceToHost));
    float result = out;
    MUST(cudaFree(device));
    return result;
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting reciprocal
*/
static void
__hoisted_reciprocal_0(float x, float *device)
{
    *device = kpr_f32_rcp_approx_ftz(x);
}

float Kuiper_Example_Float32GPU_reciprocal(float x)
{
    float out = (float) 0LL;
    float *device = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_reciprocal_0, 1U, 1U, 0U, s, x, device);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&out, device, sizeof(float), cudaMemcpyDeviceToHost));
    float result = out;
    MUST(cudaFree(device));
    return result;
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting inverse_root
*/
static void
__hoisted_inverse_root_0(float x, float *device)
{
    *device = kpr_f32_rsqrt_approx_ftz(x);
}

float Kuiper_Example_Float32GPU_inverse_root(float x)
{
    float out = (float) 0LL;
    float *device = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_inverse_root_0, 1U, 1U, 0U, s, x, device);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&out, device, sizeof(float), cudaMemcpyDeviceToHost));
    float result = out;
    MUST(cudaFree(device));
    return result;
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting arithmetic
*/
static void
__hoisted_arithmetic_0(uint32_t n, float *inputs, float *outputs)
{
    uint32_t i = 0U;
    for (; i < n; i++) {
        uint32_t j = i;
        float x = inputs[3U * j];
        float y = inputs[3U * j + 1U];
        float z = inputs[3U * j + 2U];
        float sum = kpr_f32_add_rn_ftz(x, y);
        float fused = kpr_f32_fma_rn_ftz(x, y, z);
        outputs[2U * j] = sum;
        outputs[2U * j + 1U] = fused;
    }
}

void Kuiper_Example_Float32GPU_arithmetic(
    uint32_t n, float *inputs, float *outputs)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_arithmetic_0, 1U, 1U, 0U, s, n, inputs, outputs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
