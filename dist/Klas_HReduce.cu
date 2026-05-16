
#include "Klas_HReduce.h"

__global__
/**
  hoisted when extracting reduce_f16_plus
*/
static void __hoisted_0(uint32_t nth, uint32_t lena, half *a, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc = __hadd(acc, a[idx]);
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

half Klas_HReduce_reduce_f16_plus(uint32_t nth, uint32_t lena, half *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    half *out = (half *) KPR_GPU_ALLOC(sizeof((half) 0), 1U);
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 2U * nth));
    KPR_KCALL(__hoisted_0, 1U, nth, 2U * nth, nth, lena, a, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, sizeof((half) 0), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_f32_plus
*/
static void __hoisted_1(uint32_t nth, uint32_t lena, float *a, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += a[idx];
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

float Klas_HReduce_reduce_f32_plus(uint32_t nth, uint32_t lena, float *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    float *out = (float *)KPR_GPU_ALLOC(sizeof((float) 0), 1U);
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 4U * nth));
    KPR_KCALL(__hoisted_1, 1U, nth, 4U * nth, nth, lena, a, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof((float) 0), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_f64_plus
*/
static void __hoisted_2(uint32_t nth, uint32_t lena, double *a, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0l;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += a[idx];
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

double Klas_HReduce_reduce_f64_plus(uint32_t nth, uint32_t lena, double *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    double *out = (double *)KPR_GPU_ALLOC(sizeof((double) 0), 1U);
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8U * nth));
    KPR_KCALL(__hoisted_2, 1U, nth, 8U * nth, nth, lena, a, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0l;
    MUST(cudaMemcpy(&hout, out, sizeof((double) 0), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_u32_plus
*/
static void __hoisted_3(uint32_t nth, uint32_t lena, uint32_t *a, uint32_t *out)
{
    uint32_t *sa = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t acc = 0U;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += a[idx];
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

uint32_t Klas_HReduce_reduce_u32_plus(uint32_t nth, uint32_t lena, uint32_t *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    uint32_t *out = (uint32_t *) KPR_GPU_ALLOC(sizeof((uint32_t) 0), 1U);
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 4U * nth));
    KPR_KCALL(__hoisted_3, 1U, nth, 4U * nth, nth, lena, a, out);
    MUST(cudaDeviceSynchronize());
    uint32_t hout = 0U;
    MUST(cudaMemcpy(&hout, out, sizeof((uint32_t) 0), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_u64_plus
*/
static void __hoisted_4(uint32_t nth, uint32_t lena, uint64_t *a, uint64_t *out)
{
    uint64_t *sa = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t acc = 0ULL;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth)
        acc += a[idx];
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

uint64_t Klas_HReduce_reduce_u64_plus(uint32_t nth, uint32_t lena, uint64_t *a)
{
    uint64_t lena64 = (uint64_t) lena;
    KPR_ASSERT(!(lena64 + (uint64_t) nth < lena64));
    uint64_t *out = (uint64_t *) KPR_GPU_ALLOC(sizeof((uint64_t) 0), 1U);
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 8U * nth));
    KPR_KCALL(__hoisted_4, 1U, nth, 8U * nth, nth, lena, a, out);
    MUST(cudaDeviceSynchronize());
    uint64_t hout = 0ULL;
    MUST(cudaMemcpy(&hout, out, sizeof((uint64_t) 0), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}
