
#include "Klas_AtomicReduce.h"

__global__
/**
  hoisted when extracting reduce_u32
*/
static void __hoisted_0(uint32_t *a, uint32_t *gr)
{
    atomic_add_u32(gr, a[blockIdx.x]);
}

uint32_t Klas_AtomicReduce_reduce_u32(uint32_t n, uint32_t *a)
{
    uint32_t r = 0U;
    uint32_t *gr = (uint32_t *) KPR_GPU_ALLOC(sizeof(uint32_t), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(uint32_t), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

__global__
/**
  hoisted when extracting reduce_u64
*/
static void __hoisted_1(uint64_t *a, uint64_t *gr)
{
    atomic_add_u64(gr, a[blockIdx.x]);
}

uint64_t Klas_AtomicReduce_reduce_u64(uint32_t n, uint64_t *a)
{
    uint64_t r = 0ULL;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(uint64_t), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_1, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

__global__
/**
  hoisted when extracting reduce_f32
*/
static void __hoisted_2(float *a, float *gr)
{
    atomic_add_f32(gr, a[blockIdx.x]);
}

float Klas_AtomicReduce_reduce_f32(uint32_t n, float *a)
{
    float r = 0.0f;
    float *gr = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(float), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_2, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

__global__
/**
  hoisted when extracting reduce_f64
*/
static void __hoisted_3(double *a, double *gr)
{
    atomic_add_f64(gr, a[blockIdx.x]);
}

double Klas_AtomicReduce_reduce_f64(uint32_t n, double *a)
{
    double r = 0.0l;
    double *gr = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    MUST(cudaMemcpy(gr, &r, sizeof(double), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_3, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}
