

#include "Kuiper_AtomicReduce.h"

__global__
/**
  hoisted when extracting reduce_u32
*/
static void __hoisted_0(uint32_t *a, uint32_t *gr)
{
  atomic_add_u32(gr, a[blockIdx.x]);
}

uint32_t Kuiper_AtomicReduce_reduce_u32(size_t n, uint32_t *a)
{
  uint32_t r = 0U;
  uint32_t *gr = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, (size_t)1U);
  MUST(cudaMemcpy(gr, &r, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0, n, (size_t)1U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)4U, cudaMemcpyDeviceToHost));
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

uint64_t Kuiper_AtomicReduce_reduce_u64(size_t n, uint64_t *a)
{
  uint64_t r = 0ULL;
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, (size_t)1U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_1, n, (size_t)1U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
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

float_t Kuiper_AtomicReduce_reduce_f32(size_t n, float_t *a)
{
  float_t r = (float_t)0.0f;
  float_t *gr = (float_t *)KPR_GPU_ALLOC((size_t)4U, (size_t)1U);
  MUST(cudaMemcpy(gr, &r, (size_t)4U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_2, n, (size_t)1U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)4U, cudaMemcpyDeviceToHost));
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

double_t Kuiper_AtomicReduce_reduce_f64(size_t n, double_t *a)
{
  double_t r = (double_t)0.0l;
  double_t *gr = (double_t *)KPR_GPU_ALLOC((size_t)8U, (size_t)1U);
  MUST(cudaMemcpy(gr, &r, (size_t)8U, cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_3, n, (size_t)1U, (size_t)0U, a, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(&r, gr, (size_t)8U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gr));
  return r;
}

