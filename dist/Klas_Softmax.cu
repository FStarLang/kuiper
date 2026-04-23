
#include "Klas_Softmax.h"

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_0(uint32_t lena, half *ga)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            hexp(ga[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_1(uint32_t nth, uint32_t lena, half *a_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc = __hadd(acc, a_[idx]);
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
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_2(uint32_t lena, half *ga, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(ga[1024U * blockIdx.x + threadIdx.x], sum);
}

void Klas_Softmax_softmax_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(2U, lena);
    MUST(cudaMemcpy(ga, a, 2U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga);
    MUST(cudaDeviceSynchronize());
    half *a_ = (half *) KPR_GPU_ALLOC(2U, lena);
    MUST(cudaMemcpy(a_, ga, 2U * lena, cudaMemcpyDeviceToDevice));
    half *out = (half *) KPR_GPU_ALLOC(2U, 1U);
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 2U * nth));
    KPR_KCALL(__hoisted_1, 1U, nth, 2U * nth, nth, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, 2U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_2, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 2U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_3(uint32_t lena, float *ga)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            expf(ga[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_4(uint32_t nth, uint32_t lena, float *a_, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += a_[idx];
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
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_5(uint32_t lena, float *ga, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] /= sum;
}

void Klas_Softmax_softmax_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(ga, a, 4U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_3, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga);
    MUST(cudaDeviceSynchronize());
    float *a_ = (float *)KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(a_, ga, 4U * lena, cudaMemcpyDeviceToDevice));
    float *out = (float *)KPR_GPU_ALLOC(4U, 1U);
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 4U * nth));
    KPR_KCALL(__hoisted_4, 1U, nth, 4U * nth, nth, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_5, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 4U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_6(uint32_t lena, double *ga)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            exp(ga[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_7(uint32_t nth, uint32_t lena, double *a_, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0l;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += a_[idx];
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
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_8(uint32_t lena, double *ga, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] /= sum;
}

void Klas_Softmax_softmax_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(ga, a, 8U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_6, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga);
    MUST(cudaDeviceSynchronize());
    double *a_ = (double *)KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(a_, ga, 8U * lena, cudaMemcpyDeviceToDevice));
    double *out = (double *)KPR_GPU_ALLOC(8U, 1U);
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 8U * nth));
    KPR_KCALL(__hoisted_7, 1U, nth, 8U * nth, nth, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0l;
    MUST(cudaMemcpy(&hout, out, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_8, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 8U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_9(uint32_t lena, half *ga)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            hexp(ga[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_10(uint32_t lena, half *a_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc = __hadd(acc, a_[idx]);
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
  hoisted when extracting softmax_f16
*/
static void __hoisted_11(uint32_t lena, half *ga, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(ga[1024U * blockIdx.x + threadIdx.x], sum);
}

void Klas_Softmax_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(2U, lena);
    MUST(cudaMemcpy(ga, a, 2U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_9, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga);
    MUST(cudaDeviceSynchronize());
    half *a_ = (half *) KPR_GPU_ALLOC(2U, lena);
    MUST(cudaMemcpy(a_, ga, 2U * lena, cudaMemcpyDeviceToDevice));
    half *out = (half *) KPR_GPU_ALLOC(2U, 1U);
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_10, 1U, 1024U, 2048U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, 2U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_11, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 2U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_12(uint32_t lena, float *ga)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            expf(ga[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_13(uint32_t lena, float *a_, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc += a_[idx];
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
  hoisted when extracting softmax_f32
*/
static void __hoisted_14(uint32_t lena, float *ga, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] /= sum;
}

void Klas_Softmax_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(ga, a, 4U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_12, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga);
    MUST(cudaDeviceSynchronize());
    float *a_ = (float *)KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(a_, ga, 4U * lena, cudaMemcpyDeviceToDevice));
    float *out = (float *)KPR_GPU_ALLOC(4U, 1U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_13, 1U, 1024U, 4096U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_14, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 4U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_15(uint32_t lena, double *ga)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            exp(ga[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_16(uint32_t lena, double *a_, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0l;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc += a_[idx];
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
  hoisted when extracting softmax_f64
*/
static void __hoisted_17(uint32_t lena, double *ga, double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] /= sum;
}

void Klas_Softmax_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(ga, a, 8U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_15, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga);
    MUST(cudaDeviceSynchronize());
    double *a_ = (double *)KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(a_, ga, 8U * lena, cudaMemcpyDeviceToDevice));
    double *out = (double *)KPR_GPU_ALLOC(8U, 1U);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_16, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_16, 1U, 1024U, 8192U, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0l;
    MUST(cudaMemcpy(&hout, out, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_17, (lena + 1023U) / 1024U, 1024U, 0U, lena, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 8U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
