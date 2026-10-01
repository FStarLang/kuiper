
#include "Klas_Softmax.h"

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void
__hoisted_softmax_gpu_n_f16_0(
    half *a_, uint32_t nthm, uint32_t lena, half *maxs)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = kpr_hfmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void
__hoisted_softmax_gpu_n_f16_1(uint32_t lena, half *maxs, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void
__hoisted_softmax_gpu_n_f16_2(uint32_t lena, half *a_, uint32_t nth, half *sums)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc = __hadd(acc, hexp(a_[idx]));
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void
__hoisted_softmax_gpu_n_f16_3(uint32_t lena, half *sums, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = __hdiv(hexp(a_[col]), va);
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void
__hoisted_softmax_gpu_n_f32_0(
    float *a_, uint32_t nthm, uint32_t lena, float *maxs)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmaxf(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void
__hoisted_softmax_gpu_n_f32_1(uint32_t lena, float *maxs, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void
__hoisted_softmax_gpu_n_f32_2(
    uint32_t lena, float *a_, uint32_t nth, float *sums)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += expf(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void
__hoisted_softmax_gpu_n_f32_3(uint32_t lena, float *sums, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = expf(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void
__hoisted_softmax_gpu_n_f64_0(
    double *a_, uint32_t nthm, uint32_t lena, double *maxs)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void
__hoisted_softmax_gpu_n_f64_1(uint32_t lena, double *maxs, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void
__hoisted_softmax_gpu_n_f64_2(
    uint32_t lena, double *a_, uint32_t nth, double *sums)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += exp(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void
__hoisted_softmax_gpu_n_f64_3(uint32_t lena, double *sums, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = exp(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void
__hoisted_softmax_gpu_f16_0(half *a_, uint32_t nthm, uint32_t lena, half *maxs)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = kpr_hfmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f16
*/
static void
__hoisted_softmax_gpu_f16_1(uint32_t lena, half *maxs, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f16
*/
static void
__hoisted_softmax_gpu_f16_2(uint32_t lena, half *a_, half *sums)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc = __hadd(acc, hexp(a_[idx]));
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f16
*/
static void
__hoisted_softmax_gpu_f16_3(uint32_t lena, half *sums, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = __hdiv(hexp(a_[col]), va);
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void
__hoisted_softmax_gpu_f32_0(
    float *a_, uint32_t nthm, uint32_t lena, float *maxs)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmaxf(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f32
*/
static void
__hoisted_softmax_gpu_f32_1(uint32_t lena, float *maxs, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f32
*/
static void
__hoisted_softmax_gpu_f32_2(uint32_t lena, float *a_, float *sums)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc += expf(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f32
*/
static void
__hoisted_softmax_gpu_f32_3(uint32_t lena, float *sums, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = expf(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void
__hoisted_softmax_gpu_f64_0(
    double *a_, uint32_t nthm, uint32_t lena, double *maxs)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f64
*/
static void
__hoisted_softmax_gpu_f64_1(uint32_t lena, double *maxs, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f64
*/
static void
__hoisted_softmax_gpu_f64_2(uint32_t lena, double *a_, double *sums)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc += exp(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_gpu_f64
*/
static void
__hoisted_softmax_gpu_f64_3(uint32_t lena, double *sums, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = exp(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void
__hoisted_softmax_n_f16_0(half *a_, uint32_t nthm, uint32_t lena, half *maxs)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = kpr_hfmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_n_f16
*/
static void
__hoisted_softmax_n_f16_1(uint32_t lena, half *maxs, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void
__hoisted_softmax_n_f16_2(uint32_t lena, half *a_, uint32_t nth, half *sums)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc = __hadd(acc, hexp(a_[idx]));
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_n_f16
*/
static void
__hoisted_softmax_n_f16_3(uint32_t lena, half *sums, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = __hdiv(hexp(a_[col]), va);
    }
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void
__hoisted_softmax_n_f32_0(float *a_, uint32_t nthm, uint32_t lena, float *maxs)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmaxf(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_n_f32
*/
static void
__hoisted_softmax_n_f32_1(uint32_t lena, float *maxs, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void
__hoisted_softmax_n_f32_2(uint32_t lena, float *a_, uint32_t nth, float *sums)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += expf(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_n_f32
*/
static void
__hoisted_softmax_n_f32_3(uint32_t lena, float *sums, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = expf(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void
__hoisted_softmax_n_f64_0(
    double *a_, uint32_t nthm, uint32_t lena, double *maxs)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_n_f64
*/
static void
__hoisted_softmax_n_f64_1(uint32_t lena, double *maxs, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void
__hoisted_softmax_n_f64_2(uint32_t lena, double *a_, uint32_t nth, double *sums)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += exp(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_n_f64
*/
static void
__hoisted_softmax_n_f64_3(uint32_t lena, double *sums, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = exp(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void
__hoisted_softmax_f16_0(half *a_, uint32_t nthm, uint32_t lena, half *maxs)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = kpr_hfmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f16
*/
static void
__hoisted_softmax_f16_1(uint32_t lena, half *maxs, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f16
*/
static void
__hoisted_softmax_f16_2(uint32_t lena, half *a_, half *sums)
{
    half *gsa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc = __hadd(acc, hexp(a_[idx]));
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f16
*/
static void
__hoisted_softmax_f16_3(uint32_t lena, half *sums, half *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = __hdiv(hexp(a_[col]), va);
    }
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void
__hoisted_softmax_f32_0(float *a_, uint32_t nthm, uint32_t lena, float *maxs)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmaxf(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f32
*/
static void
__hoisted_softmax_f32_1(uint32_t lena, float *maxs, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f32
*/
static void
__hoisted_softmax_f32_2(uint32_t lena, float *a_, float *sums)
{
    float *gsa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc += expf(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f32
*/
static void
__hoisted_softmax_f32_3(uint32_t lena, float *sums, float *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = expf(a_[col]) / va;
    }
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void
__hoisted_softmax_f64_0(double *a_, uint32_t nthm, uint32_t lena, double *maxs)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    gsa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < nthm)
            if ((threadIdx.x &
                    (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) - 1U) == 0U)
                gsa[threadIdx.x] = fmax(gsa[threadIdx.x], gsa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f64
*/
static void
__hoisted_softmax_f64_1(uint32_t lena, double *maxs, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f64
*/
static void
__hoisted_softmax_f64_2(uint32_t lena, double *a_, double *sums)
{
    double *gsa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U)
        acc += exp(a_[idx]);
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
        sums[blockIdx.x] = *gsa;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting softmax_f64
*/
static void
__hoisted_softmax_f64_3(uint32_t lena, double *sums, double *a_)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        a_[col] = exp(a_[col]) / va;
    }
}

void Klas_Softmax_softmax_gpu_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *maxs = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nthm);
    if ((uint32_t) sizeof(half) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_0, 1U, nthm,
        (uint32_t) sizeof(half) * nthm, s, a, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nth);
    if ((uint32_t) sizeof(half) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f16_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_2, 1U, nth,
        (uint32_t) sizeof(half) * nth, s2, lena, a, nth, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, a);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

void Klas_Softmax_softmax_gpu_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *maxs = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    float *sums = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nthm);
    if ((uint32_t) sizeof(float) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_0, 1U, nthm,
        (uint32_t) sizeof(float) * nthm, s, a, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nth);
    if ((uint32_t) sizeof(float) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f32_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_2, 1U, nth,
        (uint32_t) sizeof(float) * nth, s2, lena, a, nth, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, a);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

void Klas_Softmax_softmax_gpu_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *maxs = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    double *sums = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nthm);
    if ((uint32_t) sizeof(double) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_0, 1U, nthm,
        (uint32_t) sizeof(double) * nthm, s, a, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nth);
    if ((uint32_t) sizeof(double) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f64_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_2, 1U, nth,
        (uint32_t) sizeof(double) * nth, s2, lena, a, nth, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, a);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

void Klas_Softmax_softmax_gpu_f16(uint32_t lena, half *a)
{
    half *maxs = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nthm);
    if ((uint32_t) sizeof(half) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f16_0, 1U, nthm,
        (uint32_t) sizeof(half) * nthm, s, a, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * 1024U);
    if ((uint32_t) sizeof(half) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f16_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 1024U));
    KPR_KCALL(__hoisted_softmax_gpu_f16_2, 1U, 1024U,
        (uint32_t) sizeof(half) * 1024U, s2, lena, a, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f16_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, a);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

void Klas_Softmax_softmax_gpu_f32(uint32_t lena, float *a)
{
    float *maxs = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    float *sums = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nthm);
    if ((uint32_t) sizeof(float) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f32_0, 1U, nthm,
        (uint32_t) sizeof(float) * nthm, s, a, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f32_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_softmax_gpu_f32_2, 1U, 1024U,
        (uint32_t) sizeof(float) * 1024U, s2, lena, a, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f32_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, a);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

void Klas_Softmax_softmax_gpu_f64(uint32_t lena, double *a)
{
    double *maxs = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    double *sums = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nthm);
    if ((uint32_t) sizeof(double) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f64_0, 1U, nthm,
        (uint32_t) sizeof(double) * nthm, s, a, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * 1024U);
    if ((uint32_t) sizeof(double) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f64_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 1024U));
    KPR_KCALL(__hoisted_softmax_gpu_f64_2, 1U, 1024U,
        (uint32_t) sizeof(double) * 1024U, s2, lena, a, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f64_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, a);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

void Klas_Softmax_softmax_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *maxs = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nthm);
    if ((uint32_t) sizeof(half) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nthm));
    KPR_KCALL(__hoisted_softmax_n_f16_0, 1U, nthm,
        (uint32_t) sizeof(half) * nthm, s, ga, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, ga);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nth);
    if ((uint32_t) sizeof(half) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f16_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nth));
    KPR_KCALL(__hoisted_softmax_n_f16_2, 1U, nth, (uint32_t) sizeof(half) * nth,
        s2, lena, ga, nth, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f16_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, ga);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_Softmax_softmax_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *ga = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *maxs = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    float *sums = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nthm);
    if ((uint32_t) sizeof(float) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nthm));
    KPR_KCALL(__hoisted_softmax_n_f32_0, 1U, nthm,
        (uint32_t) sizeof(float) * nthm, s, ga, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, ga);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nth);
    if ((uint32_t) sizeof(float) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f32_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nth));
    KPR_KCALL(__hoisted_softmax_n_f32_2, 1U, nth,
        (uint32_t) sizeof(float) * nth, s2, lena, ga, nth, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f32_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, ga);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_Softmax_softmax_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *ga = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *maxs = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    double *sums = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nthm);
    if ((uint32_t) sizeof(double) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nthm));
    KPR_KCALL(__hoisted_softmax_n_f64_0, 1U, nthm,
        (uint32_t) sizeof(double) * nthm, s, ga, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, ga);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nth);
    if ((uint32_t) sizeof(double) * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f64_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nth));
    KPR_KCALL(__hoisted_softmax_n_f64_2, 1U, nth,
        (uint32_t) sizeof(double) * nth, s2, lena, ga, nth, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f64_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, ga);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_Softmax_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *maxs = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC((uint32_t) sizeof(half), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * nthm);
    if ((uint32_t) sizeof(half) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_f16_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * nthm));
    KPR_KCALL(__hoisted_softmax_f16_0, 1U, nthm, (uint32_t) sizeof(half) * nthm,
        s, ga, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f16_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, ga);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(half) * 1024U);
    if ((uint32_t) sizeof(half) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_f16_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(half) * 1024U));
    KPR_KCALL(__hoisted_softmax_f16_2, 1U, 1024U,
        (uint32_t) sizeof(half) * 1024U, s2, lena, ga, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f16_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, ga);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_Softmax_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *maxs = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    float *sums = (float *) KPR_GPU_ALLOC((uint32_t) sizeof(float), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * nthm);
    if ((uint32_t) sizeof(float) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_f32_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * nthm));
    KPR_KCALL(__hoisted_softmax_f32_0, 1U, nthm,
        (uint32_t) sizeof(float) * nthm, s, ga, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f32_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, ga);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_f32_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_softmax_f32_2, 1U, 1024U,
        (uint32_t) sizeof(float) * 1024U, s2, lena, ga, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f32_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, ga);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

void Klas_Softmax_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), lena);
    MUST(cudaMemcpy(
        ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *maxs = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    double *sums = (double *) KPR_GPU_ALLOC((uint32_t) sizeof(double), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * nthm);
    if ((uint32_t) sizeof(double) * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_f64_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * nthm));
    KPR_KCALL(__hoisted_softmax_f64_0, 1U, nthm,
        (uint32_t) sizeof(double) * nthm, s, ga, nthm, lena, maxs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f64_1,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, lena,
        maxs, ga);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * 1024U);
    if ((uint32_t) sizeof(double) * 1024U >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_f64_2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 1024U));
    KPR_KCALL(__hoisted_softmax_f64_2, 1U, 1024U,
        (uint32_t) sizeof(double) * 1024U, s2, lena, ga, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    cudaStream_t s3 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f64_3,
        lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s3, lena,
        sums, ga);
    MUST(cudaStreamSynchronize(s3));
    MUST(cudaStreamDestroy(s3));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy(
        a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
