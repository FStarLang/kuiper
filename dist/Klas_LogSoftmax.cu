
#include "Klas_LogSoftmax.h"

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f16
*/
static void __hoisted_log_softmax_gpu_n_f16_0(uint32_t nth, uint32_t lena,
                                              half *a_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        half v_ = hexp(a_[idx]);
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f16
*/
static void __hoisted_log_softmax_gpu_n_f16_1(uint32_t lena, half *a, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

void Klas_LogSoftmax_log_softmax_gpu_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *a_ = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (a_, a, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToDevice));
    half *out = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_n_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f16_0, 1U, nth, 2U * nth, nth, lena,
              a_, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f32
*/
static void
__hoisted_log_softmax_gpu_n_f32_0(uint32_t nth, uint32_t lena, float *a_,
                                  float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        float v_ = expf(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f32
*/
static void __hoisted_log_softmax_gpu_n_f32_1(uint32_t lena, float *a,
                                              float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

void Klas_LogSoftmax_log_softmax_gpu_n_f32(uint32_t nth, uint32_t lena,
                                           float *a)
{
    float *a_ = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (a_, a, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToDevice));
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_n_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f32_0, 1U, nth, 4U * nth, nth, lena,
              a_, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f64
*/
static void
__hoisted_log_softmax_gpu_n_f64_0(uint32_t nth, uint32_t lena, double *a_,
                                  double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        double v_ = exp(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_gpu_n_f64
*/
static void __hoisted_log_softmax_gpu_n_f64_1(uint32_t lena, double *a,
                                              double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

void Klas_LogSoftmax_log_softmax_gpu_n_f64(uint32_t nth, uint32_t lena,
                                           double *a)
{
    double *a_ = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (a_, a, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToDevice));
    double *out = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_n_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f64_0, 1U, nth, 8U * nth, nth, lena,
              a_, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_gpu_n_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting log_softmax_gpu_f16
*/
static void __hoisted_log_softmax_gpu_f16_0(uint32_t lena, half *a_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        half v_ = hexp(a_[idx]);
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_gpu_f16
*/
static void __hoisted_log_softmax_gpu_f16_1(uint32_t lena, half *a, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

void Klas_LogSoftmax_log_softmax_gpu_f16(uint32_t lena, half *a)
{
    half *a_ = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (a_, a, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToDevice));
    half *out = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_log_softmax_gpu_f16_0, 1U, 1024U, 2048U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_gpu_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting log_softmax_gpu_f32
*/
static void __hoisted_log_softmax_gpu_f32_0(uint32_t lena, float *a_,
                                            float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v_ = expf(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_gpu_f32
*/
static void __hoisted_log_softmax_gpu_f32_1(uint32_t lena, float *a, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

void Klas_LogSoftmax_log_softmax_gpu_f32(uint32_t lena, float *a)
{
    float *a_ = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (a_, a, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToDevice));
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_log_softmax_gpu_f32_0, 1U, 1024U, 4096U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_gpu_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting log_softmax_gpu_f64
*/
static void __hoisted_log_softmax_gpu_f64_0(uint32_t lena, double *a_,
                                            double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        double v_ = exp(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_gpu_f64
*/
static void __hoisted_log_softmax_gpu_f64_1(uint32_t lena, double *a,
                                            double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

void Klas_LogSoftmax_log_softmax_gpu_f64(uint32_t lena, double *a)
{
    double *a_ = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (a_, a, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToDevice));
    double *out = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_gpu_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_log_softmax_gpu_f64_0, 1U, 1024U, 8192U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_gpu_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting log_softmax_n_f16
*/
static void __hoisted_log_softmax_n_f16_0(uint32_t nth, uint32_t lena, half *a_,
                                          half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        half v_ = hexp(a_[idx]);
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_n_f16
*/
static void __hoisted_log_softmax_n_f16_1(uint32_t lena, half *ga, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

void Klas_LogSoftmax_log_softmax_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *a_ = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (a_, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToDevice));
    half *out = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_n_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_log_softmax_n_f16_0, 1U, nth, 2U * nth, nth, lena, a_,
              out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_n_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_n_f32
*/
static void __hoisted_log_softmax_n_f32_0(uint32_t nth, uint32_t lena,
                                          float *a_, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        float v_ = expf(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_n_f32
*/
static void __hoisted_log_softmax_n_f32_1(uint32_t lena, float *ga, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

void Klas_LogSoftmax_log_softmax_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *a_ = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (a_, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToDevice));
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_n_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_log_softmax_n_f32_0, 1U, nth, 4U * nth, nth, lena, a_,
              out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_n_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_n_f64
*/
static void __hoisted_log_softmax_n_f64_0(uint32_t nth, uint32_t lena,
                                          double *a_, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        double v_ = exp(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_n_f64
*/
static void __hoisted_log_softmax_n_f64_1(uint32_t lena, double *ga, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

void Klas_LogSoftmax_log_softmax_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *a_ = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (a_, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToDevice));
    double *out = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_n_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_log_softmax_n_f64_0, 1U, nth, 8U * nth, nth, lena, a_,
              out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_n_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_f16
*/
static void __hoisted_log_softmax_f16_0(uint32_t lena, half *a_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        half v_ = hexp(a_[idx]);
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_f16
*/
static void __hoisted_log_softmax_f16_1(uint32_t lena, half *ga, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        half x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = __hsub(x, hlog(sum));
    }
}

void Klas_LogSoftmax_log_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *a_ = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (a_, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToDevice));
    half *out = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_log_softmax_f16_0, 1U, 1024U, 2048U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_f32
*/
static void __hoisted_log_softmax_f32_0(uint32_t lena, float *a_, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v_ = expf(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_f32
*/
static void __hoisted_log_softmax_f32_1(uint32_t lena, float *ga, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

void Klas_LogSoftmax_log_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *a_ = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (a_, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToDevice));
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_log_softmax_f32_0, 1U, 1024U, 4096U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_f64
*/
static void __hoisted_log_softmax_f64_0(uint32_t lena, double *a_, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        double v_ = exp(a_[idx]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting log_softmax_f64
*/
static void __hoisted_log_softmax_f64_1(uint32_t lena, double *ga, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = ga[1024U * blockIdx.x + threadIdx.x];
        ga[1024U * blockIdx.x + threadIdx.x] = x - log(sum);
    }
}

void Klas_LogSoftmax_log_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *a_ = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (a_, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToDevice));
    double *out = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_log_softmax_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_log_softmax_f64_0, 1U, 1024U, 8192U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_log_softmax_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
