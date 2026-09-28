
#include "Klas_AtomicReduce.h"

__global__ __launch_bounds__(1)
/**
  hoisted when extracting reduce_u32
*/
static void
__hoisted_reduce_u32_0(uint32_t *a, uint32_t *gr)
{
    atomic_add_u32(gr, a[blockIdx.x]);
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting reduce_u64
*/
static void
__hoisted_reduce_u64_0(uint64_t *a, uint64_t *gr)
{
    atomic_add_u64(gr, a[blockIdx.x]);
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting reduce_f32
*/
static void
__hoisted_reduce_f32_0(float *a, float *gr)
{
    atomic_add_f32(gr, a[blockIdx.x]);
}

__global__ __launch_bounds__(1)
/**
  hoisted when extracting reduce_f64
*/
static void
__hoisted_reduce_f64_0(double *a, double *gr)
{
    atomic_add_f64(gr, a[blockIdx.x]);
}

uint32_t Klas_AtomicReduce_reduce_u32(uint32_t n, uint32_t *a)
{
    uint32_t r = 0U;
    uint32_t *gr = (uint32_t *) KPR_GPU_ALLOC(4U, 1U);
    MUST(cudaMemcpy(gr, &r, 4U, cudaMemcpyHostToDevice));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_reduce_u32_0, n, 1U, 0U, s, a, gr);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&r, gr, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

uint64_t Klas_AtomicReduce_reduce_u64(uint32_t n, uint64_t *a)
{
    uint64_t r = 0ULL;
    uint64_t *gr = (uint64_t *) KPR_GPU_ALLOC(8U, 1U);
    MUST(cudaMemcpy(gr, &r, 8U, cudaMemcpyHostToDevice));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_reduce_u64_0, n, 1U, 0U, s, a, gr);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&r, gr, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

float Klas_AtomicReduce_reduce_f32(uint32_t n, float *a)
{
    float r = 0.0f;
    float *gr = (float *) KPR_GPU_ALLOC(4U, 1U);
    MUST(cudaMemcpy(gr, &r, 4U, cudaMemcpyHostToDevice));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_reduce_f32_0, n, 1U, 0U, s, a, gr);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&r, gr, 4U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}

double Klas_AtomicReduce_reduce_f64(uint32_t n, double *a)
{
    double r = 0.0;
    double *gr = (double *) KPR_GPU_ALLOC(8U, 1U);
    MUST(cudaMemcpy(gr, &r, 8U, cudaMemcpyHostToDevice));
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_reduce_f64_0, n, 1U, 0U, s, a, gr);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    MUST(cudaMemcpy(&r, gr, 8U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(gr));
    return r;
}
