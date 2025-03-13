

#include "Kuiper_MatMul_Tiled_Inst.h"

__global__

static void
__hoisted_0(
  size_t mshared,
  uint64_t *gA,
  uint64_t *gB,
  size_t mcols,
  size_t bdim,
  uint64_t *gC
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / bdim;
  size_t bcol = tid % bdim;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  size_t k = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    size_t vk = k;
    sum +=
      gA[(mrow * bdim + brow) * (mshared * bdim) + vbk * bdim + vk] *
        gB[(vbk * bdim + vk) * (mcols * bdim) + mcol * bdim + bcol];
    if (vk == bdim - (size_t)1U)
    {
      k = (size_t)0U;
      bk = vbk + (size_t)1U;
    }
    else
      k = vk + (size_t)1U;
  }
  gC[(bi * bdim + i) * (mcols * bdim) + bj * bdim + j] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_Inst_matmul_u64_rrr(
  size_t bdim,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  size_t mcols = cols / bdim;
  size_t mshared = shared / bdim;
  size_t mrows = rows / bdim;
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * bdim * mshared * bdim));
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mshared * bdim * mcols * bdim));
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * bdim * mcols * bdim));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (mshared * bdim * (mcols * bdim)), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gA, a, (size_t)8U * (mrows * bdim * (mshared * bdim)), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0,
    mrows * mcols,
    bdim * bdim,
    (size_t)4U,
    (size_t)0U,
    mshared,
    gA,
    gB,
    mcols,
    bdim,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8U * (mrows * bdim * (mcols * bdim)), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__

static void __hoisted_1(size_t mshared, uint64_t *gA, uint64_t *gB, size_t mcols, uint64_t *gC)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  size_t k = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    size_t vk = k;
    sum +=
      gA[(mrow * (size_t)32U + brow) * (mshared * (size_t)32U) + vbk * (size_t)32U + vk] *
        gB[(vbk * (size_t)32U + vk) * (mcols * (size_t)32U) + mcol * (size_t)32U + bcol];
    if (vk == (size_t)31U)
    {
      k = (size_t)0U;
      bk = vbk + (size_t)1U;
    }
    else
      k = vk + (size_t)1U;
  }
  gC[(bi * (size_t)32U + i) * (mcols * (size_t)32U) + bj * (size_t)32U + j] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_Inst_matmul_u64_rrr_tile32(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  size_t mcols = cols / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mrows = rows / (size_t)32U;
  uint64_t
  *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * (size_t)32U * mshared * (size_t)32U));
  uint64_t
  *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mshared * (size_t)32U * mcols * (size_t)32U));
  uint64_t
  *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * (size_t)32U * mcols * (size_t)32U));
  MUST(cudaMemcpy(gB,
      b,
      (size_t)8U * (mshared * (size_t)32U * (mcols * (size_t)32U)),
      cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gA,
      a,
      (size_t)8U * (mrows * (size_t)32U * (mshared * (size_t)32U)),
      cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_1,
    mrows * mcols,
    (size_t)1024U,
    (size_t)4U,
    (size_t)0U,
    mshared,
    gA,
    gB,
    mcols,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c,
      gC,
      (size_t)8U * (mrows * (size_t)32U * (mcols * (size_t)32U)),
      cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

