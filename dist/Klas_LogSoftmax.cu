
#include "Klas_LogSoftmax.h"

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f16
*/
static void
__hoisted_log_softmax_gpu_n_f16_0(
    uint32_t lena, half *x_, uint32_t nth, half *out)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        half v_ = hexp(x_[idx]);
        acc = __hadd(acc, v_);
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nth)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = __hadd(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_n_f16
*/
static void
__hoisted_log_softmax_gpu_n_f16_1(uint32_t lena, half *a, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f32
*/
static void
__hoisted_log_softmax_gpu_n_f32_0(
    uint32_t lena, float *x_, uint32_t nth, float *out)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        float v_ = expf(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nth)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_n_f32
*/
static void
__hoisted_log_softmax_gpu_n_f32_1(uint32_t lena, float *a, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f64
*/
static void
__hoisted_log_softmax_gpu_n_f64_0(
    uint32_t lena, double *x_, uint32_t nth, double *out)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        double v_ = exp(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nth)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_n_f64
*/
static void
__hoisted_log_softmax_gpu_n_f64_1(uint32_t lena, double *a, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_f16
*/
static void
__hoisted_log_softmax_gpu_f16_0(uint32_t lena, half *x_, half *out)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        half v_ = hexp(x_[idx]);
        acc = __hadd(acc, v_);
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < 1024U)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = __hadd(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_f16
*/
static void
__hoisted_log_softmax_gpu_f16_1(uint32_t lena, half *a, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_f32
*/
static void
__hoisted_log_softmax_gpu_f32_0(uint32_t lena, float *x_, float *out)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v_ = expf(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < 1024U)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_f32
*/
static void
__hoisted_log_softmax_gpu_f32_1(uint32_t lena, float *a, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_f64
*/
static void
__hoisted_log_softmax_gpu_f64_0(uint32_t lena, double *x_, double *out)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        double v_ = exp(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < 1024U)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_gpu_f64
*/
static void
__hoisted_log_softmax_gpu_f64_1(uint32_t lena, double *a, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

__global__
/**
  hoisted when extracting log_softmax_n_f16
*/
static void
__hoisted_log_softmax_n_f16_0(uint32_t lena, half *x_, uint32_t nth, half *out)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        half v_ = hexp(x_[idx]);
        acc = __hadd(acc, v_);
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nth)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = __hadd(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_n_f16
*/
static void
__hoisted_log_softmax_n_f16_1(uint32_t lena, half *ga, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

__global__
/**
  hoisted when extracting log_softmax_n_f32
*/
static void
__hoisted_log_softmax_n_f32_0(
    uint32_t lena, float *x_, uint32_t nth, float *out)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        float v_ = expf(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nth)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_n_f32
*/
static void
__hoisted_log_softmax_n_f32_1(uint32_t lena, float *ga, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

__global__
/**
  hoisted when extracting log_softmax_n_f64
*/
static void
__hoisted_log_softmax_n_f64_0(
    uint32_t lena, double *x_, uint32_t nth, double *out)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        double v_ = exp(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nth)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_n_f64
*/
static void
__hoisted_log_softmax_n_f64_1(uint32_t lena, double *ga, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_f16
*/
static void
__hoisted_log_softmax_f16_0(uint32_t lena, half *x_, half *out)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        half v_ = hexp(x_[idx]);
        acc = __hadd(acc, v_);
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < 1024U)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = __hadd(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_f16
*/
static void
__hoisted_log_softmax_f16_1(uint32_t lena, half *ga, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_f32
*/
static void
__hoisted_log_softmax_f32_0(uint32_t lena, float *x_, float *out)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v_ = expf(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < 1024U)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_f32
*/
static void
__hoisted_log_softmax_f32_1(uint32_t lena, float *ga, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_f64
*/
static void
__hoisted_log_softmax_f64_0(uint32_t lena, double *x_, double *out)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        double v_ = exp(x_[idx]);
        acc += v_;
    }
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < 1024U)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] += gsa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting log_softmax_f64
*/
static void
__hoisted_log_softmax_f64_1(uint32_t lena, double *ga, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

void Klas_LogSoftmax_log_softmax_gpu_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *out0 = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nth);
    if ((uint32_t) sizeof(half) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_n_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nth));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f16_0, 1U, nth,
        (uint32_t) sizeof(half) * nth, s, lena, a, nth, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, a,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void Klas_LogSoftmax_log_softmax_gpu_n_f32(
    uint32_t nth, uint32_t lena, float *a)
{
    float *out0 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nth);
    if ((uint32_t) sizeof(float) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_n_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nth));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f32_0, 1U, nth,
        (uint32_t) sizeof(float) * nth, s, lena, a, nth, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float *local_out = (float *) KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, a,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void Klas_LogSoftmax_log_softmax_gpu_n_f64(
    uint32_t nth, uint32_t lena, double *a)
{
    double *out0 = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nth);
    if ((uint32_t) sizeof(double) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_n_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nth));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f64_0, 1U, nth,
        (uint32_t) sizeof(double) * nth, s, lena, a, nth, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    double *local_out = (double *) KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, a,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void Klas_LogSoftmax_log_softmax_gpu_f16(uint32_t lena, half *a)
{
    half *out0 = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * 1024U);
    if ((uint32_t) sizeof(half) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 1024U));
    KPR_KCALL(__hoisted_log_softmax_gpu_f16_0, 1U, 1024U,
        (uint32_t) sizeof(half) * 1024U, s, lena, a, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_gpu_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, a,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void Klas_LogSoftmax_log_softmax_gpu_f32(uint32_t lena, float *a)
{
    float *out0 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_log_softmax_gpu_f32_0, 1U, 1024U,
        (uint32_t) sizeof(float) * 1024U, s, lena, a, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float *local_out = (float *) KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_gpu_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, a,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void Klas_LogSoftmax_log_softmax_gpu_f64(uint32_t lena, double *a)
{
    double *out0 = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * 1024U);
    if ((uint32_t) sizeof(double) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 1024U));
    KPR_KCALL(__hoisted_log_softmax_gpu_f64_0, 1U, 1024U,
        (uint32_t) sizeof(double) * 1024U, s, lena, a, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    double *local_out = (double *) KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_gpu_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, a,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

void Klas_LogSoftmax_log_softmax_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *out0 = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nth);
    if ((uint32_t) sizeof(half) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_n_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nth));
    KPR_KCALL(__hoisted_log_softmax_n_f16_0, 1U, nth,
        (uint32_t) sizeof(half) * nth, s, lena, ga, nth, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_n_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, ga,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_LogSoftmax_log_softmax_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *ga = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *out0 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nth);
    if ((uint32_t) sizeof(float) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_n_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nth));
    KPR_KCALL(__hoisted_log_softmax_n_f32_0, 1U, nth,
        (uint32_t) sizeof(float) * nth, s, lena, ga, nth, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float *local_out = (float *) KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_n_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, ga,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_LogSoftmax_log_softmax_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *ga = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *out0 = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nth);
    if ((uint32_t) sizeof(double) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_n_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nth));
    KPR_KCALL(__hoisted_log_softmax_n_f64_0, 1U, nth,
        (uint32_t) sizeof(double) * nth, s, lena, ga, nth, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    double *local_out = (double *) KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_n_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, ga,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_LogSoftmax_log_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *out0 = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * 1024U);
    if ((uint32_t) sizeof(half) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 1024U));
    KPR_KCALL(__hoisted_log_softmax_f16_0, 1U, 1024U,
        (uint32_t) sizeof(half) * 1024U, s, lena, ga, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, ga,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_LogSoftmax_log_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *out0 = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_log_softmax_f32_0, 1U, 1024U,
        (uint32_t) sizeof(float) * 1024U, s, lena, ga, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float *local_out = (float *) KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, ga,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_LogSoftmax_log_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *out0 = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * 1024U);
    if ((uint32_t) sizeof(double) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_log_softmax_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 1024U));
    KPR_KCALL(__hoisted_log_softmax_f64_0, 1U, 1024U,
        (uint32_t) sizeof(double) * 1024U, s, lena, ga, out0);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    double *local_out = (double *) KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy(
        local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_log_softmax_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena, ga,
        res);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
