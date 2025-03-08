

#include "Kuiper_DotProduct.h"

size_t Kuiper_DotProduct_m_size = (size_t)1024U;

size_t Kuiper_DotProduct_uint32_to_sizet(uint32_t x)
{
  return (size_t)x;
}

__global__

void Kuiper_DotProduct_kernel(uint32_t *ga1, uint32_t *ga2, uint32_t *r)
{
  size_t bid = blockIdx_x();
  size_t bdim = blockDim_x();
  size_t id = bid * bdim + threadIdx_x();
  r[id] = ga1[id] * ga2[id];
}

void Kuiper_DotProduct_main(void)
{
  KRML_CHECK_SIZE(sizeof (uint32_t), Kuiper_DotProduct_m_size);
  uint32_t *a1 = (uint32_t *)KRML_HOST_CALLOC(Kuiper_DotProduct_m_size, sizeof (uint32_t));
  KRML_CHECK_SIZE(sizeof (uint32_t), Kuiper_DotProduct_m_size);
  uint32_t *a2 = (uint32_t *)KRML_HOST_CALLOC(Kuiper_DotProduct_m_size, sizeof (uint32_t));
  KRML_CHECK_SIZE(sizeof (uint32_t), Kuiper_DotProduct_m_size);
  uint32_t *ar = (uint32_t *)KRML_HOST_CALLOC(Kuiper_DotProduct_m_size, sizeof (uint32_t));
  size_t i = (size_t)0U;
  while (i < Kuiper_DotProduct_m_size)
  {
    size_t v = i;
    a1[v] = (uint32_t)v;
    a2[v] = (uint32_t)v;
    i = v + (size_t)1U;
  }
  uint32_t *ga1 = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * Kuiper_DotProduct_m_size);
  uint32_t *ga2 = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * Kuiper_DotProduct_m_size);
  MUST(cudaMemcpy(ga1, a1, (size_t)4U * Kuiper_DotProduct_m_size, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(ga2, a2, (size_t)4U * Kuiper_DotProduct_m_size, cudaMemcpyHostToDevice));
  uint32_t *gr = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * Kuiper_DotProduct_m_size);
  KPR_KCALL_ASYNC(Kuiper_DotProduct_kernel, Kuiper_DotProduct_m_size, 1U, ga1, ga2, gr);
  cudaDeviceSynchronize();
  MUST(cudaMemcpy(ar, gr, (size_t)4U * Kuiper_DotProduct_m_size, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga1));
  MUST(cudaFree(ga2));
  MUST(cudaFree(gr));
  i = (size_t)0U;
  uint32_t psum = 0U;
  while (i < Kuiper_DotProduct_m_size)
  {
    size_t vi = i;
    psum += ar[vi];
    i = vi + (size_t)1U;
  }
  KRML_HOST_FREE(a1);
  KRML_HOST_FREE(a2);
  KRML_HOST_FREE(ar);
}

