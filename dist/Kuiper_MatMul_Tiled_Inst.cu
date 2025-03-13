

#include "Kuiper_MatMul_Tiled_Inst.h"

__global__

static void
k_u64_rrr(
  size_t bdim,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / cols;
  size_t mcol = bid % cols;
  size_t brow = tid / bdim;
  size_t bcol = tid % bdim;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  size_t k = (size_t)0U;
  while (bk < shared)
  {
    size_t vbk = bk;
    size_t vk = k;
    sum +=
      gA[(mrow * bdim + brow) * (shared * bdim) + vbk * bdim + vk] *
        gB[(vbk * bdim + vk) * (cols * bdim) + mcol * bdim + bcol];
    if (vk == bdim - (size_t)1U)
    {
      k = (size_t)0U;
      bk = vbk + (size_t)1U;
    }
    else
      k = vk + (size_t)1U;
  }
  gC[(bi * bdim + i) * (cols * bdim) + bj * bdim + j] = sum;
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
  printf("rows = %lu\n", rows);
  printf("shared = %lu\n", shared);
  printf("cols = %lu\n", cols);
  printf("bdim = %lu\n", bdim);
  size_t mcols = cols / bdim;
  size_t mshared = shared / bdim;
  size_t mrows = rows / bdim;
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * bdim * mshared * bdim));
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mshared * bdim * mcols * bdim));
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (mrows * bdim * mcols * bdim));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (mshared * bdim * (mcols * bdim)), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gA, a, (size_t)8U * (mrows * bdim * (mshared * bdim)), cudaMemcpyHostToDevice));
  printf("mrows = %lu\n", mrows);
  printf("mshared = %lu\n", mshared);
  printf("mcols = %lu\n", mcols);
  printf("bdim = %lu\n", bdim);
  KPR_KCALL(k_u64_rrr,
    mrows * mcols,
    bdim * bdim,
    (size_t)4U,
    (size_t)0U,
    bdim,
    mrows,
    mshared,
    mcols,
    gA,
    gB,
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

