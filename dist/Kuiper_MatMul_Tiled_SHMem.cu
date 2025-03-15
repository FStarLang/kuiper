

#include "Kuiper_MatMul_Tiled_SHMem.h"

__global__

static void __hoisted_2(uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  KRML_HOST_IGNORE((uint64_t *)KPR_SHMEM());
  size_t mrow = bid / (size_t)32U;
  size_t mcol = bid % (size_t)32U;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < (size_t)32U)
  {
    size_t vbk = bk;
    __syncthreads();
    uint64_t sum1 = 0ULL;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA[(mrow * (size_t)32U + brow) * (size_t)1024U + vbk * (size_t)32U + vk] *
          gB[(vbk * (size_t)32U + vk) * (size_t)1024U + mcol * (size_t)32U + bcol];
      k = vk + (size_t)1U;
    }
    uint64_t sub = sum1;
    __syncthreads();
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  gC[(bi * (size_t)32U + i) * (size_t)1024U + bj * (size_t)32U + j] = sum;
}

void Kuiper_MatMul_Tiled_SHMem_inst_gpu(uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  KPR_KCALL(__hoisted_2, (size_t)1024U, (size_t)1024U, (size_t)8U, (size_t)2048U, gA, gB, gC);
  cudaDeviceSynchronize();
}

uint64_t *Kuiper_MatMul_Tiled_SHMem_matmul(uint64_t *a, uint64_t *b)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, (size_t)1048576U);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, (size_t)1048576U);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, (size_t)1048576U);
  MUST(cudaMemcpy(gA, a, (size_t)8388608U, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8388608U, cudaMemcpyHostToDevice));
  Kuiper_MatMul_Tiled_SHMem_inst_gpu(gA, gB, gC);
  KRML_CHECK_SIZE(sizeof (uint64_t), (size_t)1048576U);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC((size_t)1048576U, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8388608U, cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

