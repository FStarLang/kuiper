
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
static void __hoisted_1(uint32_t lena, half *a_)
{
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a_[threadIdx.x] = __hadd(a_[threadIdx.x], a_[nextid]);
    }
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
    KPR_KCALL(__hoisted_1, 1U, lena, 0U, lena, a_);
    MUST(cudaDeviceSynchronize());
    half *ca = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (ca != NULL)
        *ca = 0.0f;
    MUST(cudaMemcpy(ca, a_, 2U, cudaMemcpyDeviceToHost));
    half x = *ca;
    KRML_HOST_FREE(ca);
    half sum = x;
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
static void __hoisted_4(uint32_t lena, float *a_)
{
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a_[threadIdx.x] += a_[nextid];
    }
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
    KPR_KCALL(__hoisted_4, 1U, lena, 0U, lena, a_);
    MUST(cudaDeviceSynchronize());
    float *ca = (float *)KRML_HOST_MALLOC(sizeof(float));
    if (ca != NULL)
        *ca = 0.0f;
    MUST(cudaMemcpy(ca, a_, 4U, cudaMemcpyDeviceToHost));
    float x = *ca;
    KRML_HOST_FREE(ca);
    float sum = x;
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
static void __hoisted_7(uint32_t lena, double *a_)
{
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                a_[threadIdx.x] += a_[nextid];
    }
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
    KPR_KCALL(__hoisted_7, 1U, lena, 0U, lena, a_);
    MUST(cudaDeviceSynchronize());
    double *ca = (double *)KRML_HOST_MALLOC(sizeof(double));
    if (ca != NULL)
        *ca = 0.0l;
    MUST(cudaMemcpy(ca, a_, 8U, cudaMemcpyDeviceToHost));
    double x = *ca;
    KRML_HOST_FREE(ca);
    double sum = x;
    MUST(cudaFree(a_));
    KPR_KCALL(__hoisted_8, lena, 1U, 0U, ga, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(a, ga, 8U * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
