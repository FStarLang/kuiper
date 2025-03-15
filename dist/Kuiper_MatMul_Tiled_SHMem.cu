

#include "Kuiper_MatMul_Tiled_SHMem.h"

static void fakesync(void)
{
  __syncthreads();
}

__global__

static void __hoisted_2(float_t *gA, float_t *gB, float_t *gC)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  KRML_HOST_IGNORE((float_t *)KPR_SHMEM());
  size_t mrow = bid / (size_t)32U;
  size_t mcol = bid % (size_t)32U;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < (size_t)32U)
  {
    size_t vbk = bk;
    fakesync();
    float_t sum1 = (float_t)0.0f;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA[(mrow * (size_t)32U + brow) * (size_t)1024U + vbk * (size_t)32U + vk] *
          gB[(vbk * (size_t)32U + vk) * (size_t)1024U + mcol * (size_t)32U + bcol];
      k = vk + (size_t)1U;
    }
    float_t sub = sum1;
    fakesync();
    sum += sub;
  }
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  gC[(bi * (size_t)32U + i) * (size_t)1024U + bj * (size_t)32U + j] = sum;
}

void Kuiper_MatMul_Tiled_SHMem_inst(float_t *gA, float_t *gB, float_t *gC)
{
  KPR_KCALL(__hoisted_2, (size_t)1024U, (size_t)1024U, (size_t)4U, (size_t)2048U, gA, gB, gC);
  cudaDeviceSynchronize();
}

