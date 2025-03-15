

#include "Kuiper_MatMul_Tiled.h"

__global__

static void
__hoisted_0(
  size_t tile,
  size_t mcols,
  size_t mshared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t sum1 = 0ULL;
    size_t k = (size_t)0U;
    while (k < tile)
    {
      size_t vk = k;
      sum1 +=
        gA[(mrow * tile + brow) * (mshared * tile) + vbk * tile + vk] *
          gB[(vbk * tile + vk) * (mcols * tile) + mcol * tile + bcol];
    }
    uint64_t sub = sum1;
    sum += sub;
  }
  gC[(bi * tile + i) * (mcols * tile) + bj * tile + j] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  size_t mcols = cols / tile;
  size_t mshared = shared / tile;
  size_t mrows = rows / tile;
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * tile * mshared * tile));
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mshared * tile * mcols * tile));
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * tile * mcols * tile));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (mshared * tile * (mcols * tile)), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gA, a, (size_t)8U * (mrows * tile * (mshared * tile)), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0,
    mrows * mcols,
    tile * tile,
    (size_t)1U,
    (size_t)0U,
    tile,
    mcols,
    mshared,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8U * (mrows * tile * (mcols * tile)), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__

static void __hoisted_1(size_t mcols, size_t mshared, uint64_t *gA, uint64_t *gB, uint64_t *gC)
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
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t sum1 = 0ULL;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA[(mrow * (size_t)32U + brow) * (mshared * (size_t)32U) + vbk * (size_t)32U + vk] *
          gB[(vbk * (size_t)32U + vk) * (mcols * (size_t)32U) + mcol * (size_t)32U + bcol];
    }
    uint64_t sub = sum1;
    sum += sub;
  }
  gC[(bi * (size_t)32U + i) * (mcols * (size_t)32U) + bj * (size_t)32U + j] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_rrr_tile32(
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
    (size_t)1U,
    (size_t)0U,
    mcols,
    mshared,
    gA,
    gB,
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

