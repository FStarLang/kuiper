
#include "Kuiper_AtomicReduce.h"

__global__
/**
  hoisted when extracting reduce_u32
*/
static void __hoisted_0(uint32_t *a, uint32_t *gr)
{
    atomic_add_u32(gr, a[blockIdx.x]);
}

uint32_t Kuiper_AtomicReduce_reduce_u32(uint32_t n, uint32_t *a)
{
    uint32_t r = 0U;
    uint32_t *gr = (uint32_t *) KPR_GPU_ALLOC(4U, 1U);
    MUST(cudaMemcpy(gr, &r, 4U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, 4U, cudaMemcpyDeviceToHost));
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

uint64_t Kuiper_AtomicReduce_reduce_u64(uint32_t n, uint64_t *a)
{
    uint64_t r = 0ULL;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(8U, 1U);
    MUST(cudaMemcpy(gr, &r, 8U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_1, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

__global__
/**
  hoisted when extracting reduce_f32
*/
static void __hoisted_2(float_t *a, float_t *gr)
{
    atomic_add_f32(gr, a[blockIdx.x]);
}

float_t Kuiper_AtomicReduce_reduce_f32(uint32_t n, float_t *a)
{
    float_t r = 0.0f;
    float_t *gr = (float_t *) KPR_GPU_ALLOC(4U, 1U);
    MUST(cudaMemcpy(gr, &r, 4U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_2, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

__global__
/**
  hoisted when extracting reduce_f64
*/
static void __hoisted_3(double_t *a, double_t *gr)
{
    atomic_add_f64(gr, a[blockIdx.x]);
}

double_t Kuiper_AtomicReduce_reduce_f64(uint32_t n, double_t *a)
{
    double_t r = 0.0l;
    double_t *gr = (double_t *) KPR_GPU_ALLOC(8U, 1U);
    MUST(cudaMemcpy(gr, &r, 8U, cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_3, n, 1U, 0U, a, gr);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy(&r, gr, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}
