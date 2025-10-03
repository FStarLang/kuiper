
#include "Kuiper_DotProduct.h"

__global__
/**
  hoisted when extracting dotprod_f32
*/
static void __hoisted_0(uint32_t lena, float_t *ga1, float_t *ga2)
{
    ga1[threadIdx.x] *= ga2[threadIdx.x];
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                ga1[threadIdx.x] += ga1[nextid];
    }
}

float_t Kuiper_DotProduct_dotprod_f32(uint32_t lena, float_t *a1, float_t *a2)
{
    float_t *ga1 = (float_t *) KPR_GPU_ALLOC(4U, lena);
    float_t *ga2 = (float_t *) KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(ga1, a1, 4U * lena, cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(ga2, a2, 4U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, 1U, lena, 0U, lena, ga1, ga2);
    cudaDeviceSynchronize();
    float_t *ar = (float_t *) KRML_HOST_MALLOC(sizeof(float_t));
    if (ar != NULL)
        *ar = 0.0f;
    MUST(cudaMemcpy(ar, ga1, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga1));
    MUST(cudaFree(ga2));
    float_t dp = *ar;
    KRML_HOST_FREE(ar);
    return dp;
}

__global__
/**
  hoisted when extracting dotprod_f64
*/
static void __hoisted_1(uint32_t lena, double_t *ga1, double_t *ga2)
{
    ga1[threadIdx.x] *= ga2[threadIdx.x];
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                ga1[threadIdx.x] += ga1[nextid];
    }
}

double_t Kuiper_DotProduct_dotprod_f64(uint32_t lena, double_t *a1,
                                       double_t *a2)
{
    double_t *ga1 = (double_t *) KPR_GPU_ALLOC(8U, lena);
    double_t *ga2 = (double_t *) KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(ga1, a1, 8U * lena, cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(ga2, a2, 8U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_1, 1U, lena, 0U, lena, ga1, ga2);
    cudaDeviceSynchronize();
    double_t *ar = (double_t *) KRML_HOST_MALLOC(sizeof(double_t));
    if (ar != NULL)
        *ar = 0.0l;
    MUST(cudaMemcpy(ar, ga1, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga1));
    MUST(cudaFree(ga2));
    double_t dp = *ar;
    KRML_HOST_FREE(ar);
    return dp;
}

__global__
/**
  hoisted when extracting dotprod_u32
*/
static void __hoisted_2(uint32_t lena, uint32_t *ga1, uint32_t *ga2)
{
    ga1[threadIdx.x] *= ga2[threadIdx.x];
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                ga1[threadIdx.x] += ga1[nextid];
    }
}

uint32_t Kuiper_DotProduct_dotprod_u32(uint32_t lena, uint32_t *a1,
                                       uint32_t *a2)
{
    uint32_t *ga1 = (uint32_t *) KPR_GPU_ALLOC(4U, lena);
    uint32_t *ga2 = (uint32_t *) KPR_GPU_ALLOC(4U, lena);
    MUST(cudaMemcpy(ga1, a1, 4U * lena, cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(ga2, a2, 4U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_2, 1U, lena, 0U, lena, ga1, ga2);
    cudaDeviceSynchronize();
    uint32_t *ar = (uint32_t *) KRML_HOST_CALLOC(1U, sizeof(uint32_t));
    MUST(cudaMemcpy(ar, ga1, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga1));
    MUST(cudaFree(ga2));
    uint32_t dp = *ar;
    KRML_HOST_FREE(ar);
    return dp;
}

__global__
/**
  hoisted when extracting dotprod_u64
*/
static void __hoisted_3(uint32_t lena, uint64_t *ga1, uint64_t *ga2)
{
    ga1[threadIdx.x] *= ga2[threadIdx.x];
    uint32_t n = 0U;
    for (; (uint32_t) (1U << (uint32_t) n) < lena; n++) {
        uint32_t __anf0 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf0);
        if (nextid < lena)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf0 + 1U)) -
                 1U) == 0U)
                ga1[threadIdx.x] += ga1[nextid];
    }
}

uint64_t Kuiper_DotProduct_dotprod_u64(uint32_t lena, uint64_t *a1,
                                       uint64_t *a2)
{
    uint64_t *ga1 = (uint64_t *) KPR_GPU_ALLOC(8U, lena);
    uint64_t *ga2 = (uint64_t *) KPR_GPU_ALLOC(8U, lena);
    MUST(cudaMemcpy(ga1, a1, 8U * lena, cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(ga2, a2, 8U * lena, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_3, 1U, lena, 0U, lena, ga1, ga2);
    cudaDeviceSynchronize();
    uint64_t *ar = (uint64_t *) KRML_HOST_CALLOC(1U, sizeof(uint64_t));
    MUST(cudaMemcpy(ar, ga1, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga1));
    MUST(cudaFree(ga2));
    uint64_t dp = *ar;
    KRML_HOST_FREE(ar);
    return dp;
}
