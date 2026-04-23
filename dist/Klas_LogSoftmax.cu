
#include "Klas_LogSoftmax.h"

__global__
/**
  hoisted when extracting log_softmax_f16
*/
static void __hoisted_0(half *a_)
{
    a_[blockIdx.x] = hexp(a_[blockIdx.x]);
}

__global__
/**
  hoisted when extracting log_softmax_f16
*/
static void __hoisted_1(uint32_t lena, half *a_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a_[threadIdx.x];
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < lena; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < lena)
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
static void __hoisted_2(half *ga, half sum)
{
    half x = ga[blockIdx.x];
    ga[blockIdx.x] = __hsub(x, hlog(sum));
}

void Klas_LogSoftmax_log_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(2U, lena);
    MUST(cudaMemcpy(ga, a, 2U * lena, cudaMemcpyHostToDevice));
    half *a_ = (half *) KPR_GPU_ALLOC(2U, lena);
    MUST(cudaMemcpy(a_, ga, 2U * lena, cudaMemcpyDeviceToDevice));
    KPR_KCALL(__hoisted_0, lena, 1U, 0U, a_);
    MUST(cudaDeviceSynchronize());
    half *out = (half *) KPR_GPU_ALLOC(2U, 1U);
    KPR_SHMEM_FITS(2U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 2U * lena));
    KPR_KCALL(__hoisted_1, 1U, lena, 2U * lena, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, 2U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    half sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_2, lena, 1U, 0U, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 2U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_f32
*/
static void __hoisted_3(float *a_)
{
    a_[blockIdx.x] = expf(a_[blockIdx.x]);
}

__global__
/**
  hoisted when extracting log_softmax_f32
*/
static void __hoisted_4(uint32_t lena, float *a_, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a_[threadIdx.x];
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < lena; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < lena)
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
static void __hoisted_5(float *ga, float sum)
{
    float x = ga[blockIdx.x];
    ga[blockIdx.x] = x - logf(sum);
}

void Klas_LogSoftmax_log_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(ga, a, 4U * lena, cudaMemcpyHostToDevice));
    float *a_ = (float *)KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(a_, ga, 4U * lena, cudaMemcpyDeviceToDevice));
    KPR_KCALL(__hoisted_3, lena, 1U, 0U, a_);
    MUST(cudaDeviceSynchronize());
    float *out = (float *)KPR_GPU_ALLOC(4U, 1U);
    KPR_SHMEM_FITS(4U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 4U * lena));
    KPR_KCALL(__hoisted_4, 1U, lena, 4U * lena, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_5, lena, 1U, 0U, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 4U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting log_softmax_f64
*/
static void __hoisted_6(double *a_)
{
    a_[blockIdx.x] = exp(a_[blockIdx.x]);
}

__global__
/**
  hoisted when extracting log_softmax_f64
*/
static void __hoisted_7(uint32_t lena, double *a_, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a_[threadIdx.x];
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < lena; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < lena)
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
static void __hoisted_8(double *ga, double sum)
{
    double x = ga[blockIdx.x];
    ga[blockIdx.x] = x - log(sum);
}

void Klas_LogSoftmax_log_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(ga, a, 8U * lena, cudaMemcpyHostToDevice));
    double *a_ = (double *)KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(a_, ga, 8U * lena, cudaMemcpyDeviceToDevice));
    KPR_KCALL(__hoisted_6, lena, 1U, 0U, a_);
    MUST(cudaDeviceSynchronize());
    double *out = (double *)KPR_GPU_ALLOC(8U, 1U);
    KPR_SHMEM_FITS(8U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 8U * lena));
    KPR_KCALL(__hoisted_7, 1U, lena, 8U * lena, lena, a_, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0l;
    MUST(cudaMemcpy(&hout, out, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    double sum = hout;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_8, lena, 1U, 0U, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 8U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
