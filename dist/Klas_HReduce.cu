
#include "Klas_HReduce.h"

__global__
/**
  hoisted when extracting reduce_f16_plus
*/
static void __hoisted_0(uint32_t lena, half *a, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a[threadIdx.x];
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

half Klas_HReduce_reduce_f16_plus(uint32_t lena, half *a)
{
    half *out = (half *) KPR_GPU_ALLOC(2U, 1U);
    KPR_SHMEM_FITS(2U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 2U * lena));
    KPR_KCALL(__hoisted_0, 1U, lena, 2U * lena, lena, a, out);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out, 2U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_f32_plus
*/
static void __hoisted_1(uint32_t lena, float *a, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a[threadIdx.x];
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

float Klas_HReduce_reduce_f32_plus(uint32_t lena, float *a)
{
    float *out = (float *)KPR_GPU_ALLOC(4U, 1U);
    KPR_SHMEM_FITS(4U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 4U * lena));
    KPR_KCALL(__hoisted_1, 1U, lena, 4U * lena, lena, a, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_f64_plus
*/
static void __hoisted_2(uint32_t lena, double *a, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a[threadIdx.x];
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

double Klas_HReduce_reduce_f64_plus(uint32_t lena, double *a)
{
    double *out = (double *)KPR_GPU_ALLOC(8U, 1U);
    KPR_SHMEM_FITS(8U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8U * lena));
    KPR_KCALL(__hoisted_2, 1U, lena, 8U * lena, lena, a, out);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0l;
    MUST(cudaMemcpy(&hout, out, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_u32_plus
*/
static void __hoisted_3(uint32_t lena, uint32_t *a, uint32_t *out)
{
    uint32_t *sa = (uint32_t *) KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a[threadIdx.x];
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

uint32_t Klas_HReduce_reduce_u32_plus(uint32_t lena, uint32_t *a)
{
    uint32_t *out = (uint32_t *) KPR_GPU_ALLOC(4U, 1U);
    KPR_SHMEM_FITS(4U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 4U * lena));
    KPR_KCALL(__hoisted_3, 1U, lena, 4U * lena, lena, a, out);
    MUST(cudaDeviceSynchronize());
    uint32_t hout = 0U;
    MUST(cudaMemcpy(&hout, out, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}

__global__
/**
  hoisted when extracting reduce_u64_plus
*/
static void __hoisted_4(uint32_t lena, uint64_t *a, uint64_t *out)
{
    uint64_t *sa = (uint64_t *) KPR_SHMEM_AT(0U);
    sa[threadIdx.x] = a[threadIdx.x];
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

uint64_t Klas_HReduce_reduce_u64_plus(uint32_t lena, uint64_t *a)
{
    uint64_t *out = (uint64_t *) KPR_GPU_ALLOC(8U, 1U);
    KPR_SHMEM_FITS(8U * lena);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 8U * lena));
    KPR_KCALL(__hoisted_4, 1U, lena, 8U * lena, lena, a, out);
    MUST(cudaDeviceSynchronize());
    uint64_t hout = 0ULL;
    MUST(cudaMemcpy(&hout, out, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout;
}
