
#include "Klas_HReduce.h"

__global__
/**
  hoisted when extracting reduce_f16_plus
*/
static void __hoisted_reduce_f16_plus_0(uint32_t nth, uint32_t lena, half *x_,
                                        half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc = __hadd(acc, x_[idx]);
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
        out[blockIdx.x] = *sa;
}

half Klas_HReduce_reduce_f16_plus(uint32_t nth, uint32_t lena, half *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    half *x_ = a;
    half *out0 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *out = out0;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_reduce_f16_plus_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_reduce_f16_plus_0, 1U, nth, 2U * nth, s, nth, lena, x_,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    return res;
}

__global__
/**
  hoisted when extracting reduce_f32_plus
*/
static void __hoisted_reduce_f32_plus_0(uint32_t nth, uint32_t lena, float *x_,
                                        float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += x_[idx];
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
        out[blockIdx.x] = *sa;
}

float Klas_HReduce_reduce_f32_plus(uint32_t nth, uint32_t lena, float *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    float *x_ = a;
    float *out0 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *out = out0;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_reduce_f32_plus_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_reduce_f32_plus_0, 1U, nth, 4U * nth, s, nth, lena, x_,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float *local_out = (float *)KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    return res;
}

__global__
/**
  hoisted when extracting reduce_f64_plus
*/
static void __hoisted_reduce_f64_plus_0(uint32_t nth, uint32_t lena, double *x_,
                                        double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += x_[idx];
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
        out[blockIdx.x] = *sa;
}

double Klas_HReduce_reduce_f64_plus(uint32_t nth, uint32_t lena, double *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    double *x_ = a;
    double *out0 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *out = out0;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_reduce_f64_plus_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_reduce_f64_plus_0, 1U, nth, 8U * nth, s, nth, lena, x_,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    double *local_out = (double *)KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    return res;
}

__global__
/**
  hoisted when extracting reduce_u32_plus
*/
static void
__hoisted_reduce_u32_plus_0(uint32_t nth, uint32_t lena, uint32_t *x_,
                            uint32_t *out)
{
    uint32_t *sa = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t acc = 0U;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += x_[idx];
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
        out[blockIdx.x] = *sa;
}

uint32_t Klas_HReduce_reduce_u32_plus(uint32_t nth, uint32_t lena, uint32_t *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    uint32_t *x_ = a;
    uint32_t *out0 = (uint32_t *) KPR_GPU_ALLOC(sizeof(uint32_t), 1U);
    uint32_t *out = out0;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_reduce_u32_plus_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_reduce_u32_plus_0, 1U, nth, 4U * nth, s, nth, lena, x_,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    uint32_t *local_out = (uint32_t *) KRML_HOST_CALLOC(1U, sizeof(uint32_t));
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(uint32_t),
          cudaMemcpyDeviceToHost));
    uint32_t res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    return res;
}

__global__
/**
  hoisted when extracting reduce_u64_plus
*/
static void
__hoisted_reduce_u64_plus_0(uint32_t nth, uint32_t lena, uint64_t *x_,
                            uint64_t *out)
{
    uint64_t *sa = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t acc = 0ULL;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += x_[idx];
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
        out[blockIdx.x] = *sa;
}

uint64_t Klas_HReduce_reduce_u64_plus(uint32_t nth, uint32_t lena, uint64_t *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    uint64_t *x_ = a;
    uint64_t *out0 = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 1U);
    uint64_t *out = out0;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_reduce_u64_plus_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_reduce_u64_plus_0, 1U, nth, 8U * nth, s, nth, lena, x_,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    uint64_t *local_out = (uint64_t *) KRML_HOST_CALLOC(1U, sizeof(uint64_t));
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(uint64_t),
          cudaMemcpyDeviceToHost));
    uint64_t res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    return res;
}
