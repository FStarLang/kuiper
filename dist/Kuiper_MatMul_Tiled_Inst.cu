

#include "Kuiper_MatMul_Tiled_Inst.h"

__global__

static void
__hoisted_0(
  size_t bdim,
  size_t bdim0,
  size_t mcols,
  size_t mcols0,
  size_t bdim1,
  size_t bdim2,
  size_t mshared,
  size_t bdim3,
  size_t mshared0,
  size_t bdim4,
  size_t bdim5,
  uint64_t *gA,
  size_t bdim6,
  size_t mcols1,
  size_t bdim7,
  size_t bdim8,
  uint64_t *gB,
  size_t bdim9,
  size_t bdim10,
  size_t mcols2,
  size_t bdim11,
  size_t bdim12,
  uint64_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(bdim);
  KRML_MAYBE_UNUSED_VAR(bdim0);
  KRML_MAYBE_UNUSED_VAR(mcols);
  KRML_MAYBE_UNUSED_VAR(mcols0);
  KRML_MAYBE_UNUSED_VAR(bdim1);
  KRML_MAYBE_UNUSED_VAR(bdim2);
  KRML_MAYBE_UNUSED_VAR(mshared);
  KRML_MAYBE_UNUSED_VAR(bdim3);
  KRML_MAYBE_UNUSED_VAR(bdim4);
  KRML_MAYBE_UNUSED_VAR(bdim5);
  KRML_MAYBE_UNUSED_VAR(bdim6);
  KRML_MAYBE_UNUSED_VAR(mcols1);
  KRML_MAYBE_UNUSED_VAR(bdim7);
  KRML_MAYBE_UNUSED_VAR(bdim8);
  KRML_MAYBE_UNUSED_VAR(bdim9);
  KRML_MAYBE_UNUSED_VAR(bdim10);
  KRML_MAYBE_UNUSED_VAR(bdim11);
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols2;
  size_t mcol = bid % mcols2;
  size_t brow = tid / bdim12;
  size_t bcol = tid % bdim12;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  size_t k = (size_t)0U;
  while (bk < mshared0)
  {
    size_t vbk = bk;
    size_t vk = k;
    sum +=
      gA[(mrow * bdim12 + brow) * (mshared0 * bdim12) + vbk * bdim12 + vk] *
        gB[(vbk * bdim12 + vk) * (mcols2 * bdim12) + mcol * bdim12 + bcol];
    if (vk == bdim12 - (size_t)1U)
    {
      k = (size_t)0U;
      bk = vbk + (size_t)1U;
    }
    else
      k = vk + (size_t)1U;
  }
  gC[(bi * bdim12 + i) * (mcols2 * bdim12) + bj * bdim12 + j] = sum;
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
    bdim,
    bdim,
    mcols,
    mcols,
    bdim,
    bdim,
    mshared,
    bdim,
    mshared,
    bdim,
    bdim,
    gA,
    bdim,
    mcols,
    bdim,
    bdim,
    gB,
    bdim,
    bdim,
    mcols,
    bdim,
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

static void
__hoisted_1(
  size_t mcols,
  size_t mcols0,
  size_t mshared,
  size_t mshared0,
  uint64_t *gA,
  size_t mcols1,
  uint64_t *gB,
  size_t mcols2,
  uint64_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(mcols);
  KRML_MAYBE_UNUSED_VAR(mcols0);
  KRML_MAYBE_UNUSED_VAR(mshared);
  KRML_MAYBE_UNUSED_VAR(mcols1);
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols2;
  size_t mcol = bid % mcols2;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  size_t k = (size_t)0U;
  while (bk < mshared0)
  {
    size_t vbk = bk;
    size_t vk = k;
    sum +=
      gA[(mrow * (size_t)32U + brow) * (mshared0 * (size_t)32U) + vbk * (size_t)32U + vk] *
        gB[(vbk * (size_t)32U + vk) * (mcols2 * (size_t)32U) + mcol * (size_t)32U + bcol];
    if (vk == (size_t)31U)
    {
      k = (size_t)0U;
      bk = vbk + (size_t)1U;
    }
    else
      k = vk + (size_t)1U;
  }
  gC[(bi * (size_t)32U + i) * (mcols2 * (size_t)32U) + bj * (size_t)32U + j] = sum;
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
    mcols,
    mcols,
    mshared,
    mshared,
    gA,
    mcols,
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

