
#include "Klas_Softmax.h"

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void __hoisted_softmax_gpu_n_f16_0(uint32_t lena, half *a_, half *maxs,
                                          uint32_t nthm)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void __hoisted_softmax_gpu_n_f16_1(uint32_t lena, half *a_, half *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        uint32_t ni = col;
        a_[ni] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void __hoisted_softmax_gpu_n_f16_2(uint32_t nth, uint32_t lena, half *a_,
                                          half *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void __hoisted_softmax_gpu_n_f16_3(uint32_t lena, half *a_, half *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = __hdiv(hexp(a_[col]), va1);
    }
}

void Klas_Softmax_softmax_gpu_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *a_ = a;
    half *maxs = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_0, 1U, nthm, 2U * nthm, s, lena, a_,
              maxs, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f16_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_2, 1U, nth, 2U * nth, s1, nth, lena,
              a_, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void __hoisted_softmax_gpu_n_f32_0(uint32_t lena, float *a_, float *maxs,
                                          uint32_t nthm)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void __hoisted_softmax_gpu_n_f32_1(uint32_t lena, float *a_, float *maxs)
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
static void __hoisted_softmax_gpu_n_f32_2(uint32_t nth, uint32_t lena,
                                          float *a_, float *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void __hoisted_softmax_gpu_n_f32_3(uint32_t lena, float *a_, float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = expf(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_gpu_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *a_ = a;
    float *maxs = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *sums = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_0, 1U, nthm, 4U * nthm, s, lena, a_,
              maxs, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f32_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_2, 1U, nth, 4U * nth, s1, nth, lena,
              a_, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void
__hoisted_softmax_gpu_n_f64_0(uint32_t lena, double *a_, double *maxs,
                              uint32_t nthm)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void __hoisted_softmax_gpu_n_f64_1(uint32_t lena, double *a_,
                                          double *maxs)
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
__hoisted_softmax_gpu_n_f64_2(uint32_t nth, uint32_t lena, double *a_,
                              double *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void __hoisted_softmax_gpu_n_f64_3(uint32_t lena, double *a_,
                                          double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = exp(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_gpu_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *a_ = a;
    double *maxs = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *sums = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_0, 1U, nthm, 8U * nthm, s, lena, a_,
              maxs, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f64_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_2, 1U, nth, 8U * nth, s1, nth, lena,
              a_, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_0(uint32_t lena, half *a_, half *maxs,
                                        uint32_t nthm)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_1(uint32_t lena, half *a_, half *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        uint32_t ni = col;
        a_[ni] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_2(uint32_t lena, half *a_, half *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_3(uint32_t lena, half *a_, half *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = __hdiv(hexp(a_[col]), va1);
    }
}

void Klas_Softmax_softmax_gpu_f16(uint32_t lena, half *a)
{
    half *a_ = a;
    half *maxs = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f16_0, 1U, nthm, 2U * nthm, s, lena, a_,
              maxs, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f16_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_softmax_gpu_f16_2, 1U, 1024U, 2048U, s1, lena, a_,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f16_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_0(uint32_t lena, float *a_, float *maxs,
                                        uint32_t nthm)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_1(uint32_t lena, float *a_, float *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_2(uint32_t lena, float *a_, float *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_3(uint32_t lena, float *a_, float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = expf(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_gpu_f32(uint32_t lena, float *a)
{
    float *a_ = a;
    float *maxs = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *sums = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f32_0, 1U, nthm, 4U * nthm, s, lena, a_,
              maxs, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f32_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_softmax_gpu_f32_2, 1U, 1024U, 4096U, s1, lena, a_,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f32_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_0(uint32_t lena, double *a_, double *maxs,
                                        uint32_t nthm)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_1(uint32_t lena, double *a_, double *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_2(uint32_t lena, double *a_, double *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_3(uint32_t lena, double *a_, double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = exp(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_gpu_f64(uint32_t lena, double *a)
{
    double *a_ = a;
    double *maxs = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *sums = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f64_0, 1U, nthm, 8U * nthm, s, lena, a_,
              maxs, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f64_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_softmax_gpu_f64_2, 1U, 1024U, 8192U, s1, lena, a_,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_gpu_f64_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_0(uint32_t lena, half *a_, half *maxs,
                                      uint32_t nthm)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_1(uint32_t lena, half *a_, half *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        uint32_t ni = col;
        a_[ni] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_2(uint32_t nth, uint32_t lena, half *a_,
                                      half *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_3(uint32_t lena, half *a_, half *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = __hdiv(hexp(a_[col]), va1);
    }
}

void Klas_Softmax_softmax_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *a_ = ga;
    half *maxs = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_n_f16_0, 1U, nthm, 2U * nthm, s, lena, a_, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f16_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_softmax_n_f16_2, 1U, nth, 2U * nth, s1, nth, lena, a_,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f16_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_softmax_n_f32_0(uint32_t lena, float *a_, float *maxs,
                                      uint32_t nthm)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_softmax_n_f32_1(uint32_t lena, float *a_, float *maxs)
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
static void __hoisted_softmax_n_f32_2(uint32_t nth, uint32_t lena, float *a_,
                                      float *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_softmax_n_f32_3(uint32_t lena, float *a_, float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = expf(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *a_ = ga;
    float *maxs = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *sums = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_n_f32_0, 1U, nthm, 4U * nthm, s, lena, a_, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f32_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_softmax_n_f32_2, 1U, nth, 4U * nth, s1, nth, lena, a_,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f32_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_softmax_n_f64_0(uint32_t lena, double *a_, double *maxs,
                                      uint32_t nthm)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_softmax_n_f64_1(uint32_t lena, double *a_, double *maxs)
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
static void __hoisted_softmax_n_f64_2(uint32_t nth, uint32_t lena, double *a_,
                                      double *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_softmax_n_f64_3(uint32_t lena, double *a_, double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = exp(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *a_ = ga;
    double *maxs = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *sums = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    uint32_t nthm = nth <= lena ? nth : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_n_f64_0, 1U, nthm, 8U * nthm, s, lena, a_, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f64_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_softmax_n_f64_2, 1U, nth, 8U * nth, s1, nth, lena, a_,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_n_f64_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_0(uint32_t lena, half *a_, half *maxs,
                                    uint32_t nthm)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_1(uint32_t lena, half *a_, half *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        uint32_t ni = col;
        a_[ni] =
            __hsub(a_[col], maxs[(1024U * blockIdx.x + threadIdx.x) / lena]);
    }
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_2(uint32_t lena, half *a_, half *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_3(uint32_t lena, half *a_, half *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        half va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = __hdiv(hexp(a_[col]), va1);
    }
}

void Klas_Softmax_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    half *a_ = ga;
    half *maxs = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *sums = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_f16_0, 1U, nthm, 2U * nthm, s, lena, a_, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f16_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f16_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_softmax_f16_2, 1U, 1024U, 2048U, s1, lena, a_, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f16_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_0(uint32_t lena, float *a_, float *maxs,
                                    uint32_t nthm)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_1(uint32_t lena, float *a_, float *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_2(uint32_t lena, float *a_, float *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_3(uint32_t lena, float *a_, float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        float va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = expf(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    float *a_ = ga;
    float *maxs = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *sums = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_f32_0, 1U, nthm, 4U * nthm, s, lena, a_, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f32_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_softmax_f32_2, 1U, 1024U, 4096U, s1, lena, a_, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f32_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_0(uint32_t lena, double *a_, double *maxs,
                                    uint32_t nthm)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = a_[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a_[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_1(uint32_t lena, double *a_, double *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        a_[col] -= maxs[(1024U * blockIdx.x + threadIdx.x) / lena];
    }
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_2(uint32_t lena, double *a_, double *sums)
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
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_3(uint32_t lena, double *a_, double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % lena;
        double va1 = sums[(1024U * blockIdx.x + threadIdx.x) / lena];
        uint32_t ni = col;
        a_[ni] = exp(a_[col]) / va1;
    }
}

void Klas_Softmax_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    double *a_ = ga;
    double *maxs = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *sums = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_f64_0, 1U, nthm, 8U * nthm, s, lena, a_, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f64_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s0, lena, a_, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f64_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_softmax_f64_2, 1U, 1024U, 8192U, s1, lena, a_, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_f64_3,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, s2, lena, a_, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
