

#include "Kuiper_MatMul_U64.h"

__global__

static void
k_u64_rrr(size_t rows, size_t shared, size_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  uint64_t sum = 0ULL;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[trow * shared + vi] * gB[vi * cols + tcol];
    i = vi + (size_t)1U;
  }
  gC[trow * cols + tcol] = sum;
}

uint64_t
*Kuiper_MatMul_U64_matmul_u64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (rows * shared));
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (shared * cols));
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(k_u64_rrr,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__

static void
k_u64_ccc(size_t rows, size_t shared, size_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  uint64_t sum = 0ULL;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[vi * rows + trow] * gB[tcol * shared + vi];
    i = vi + (size_t)1U;
  }
  gC[tcol * rows + trow] = sum;
}

uint64_t
*Kuiper_MatMul_U64_matmul_u64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (rows * shared));
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (shared * cols));
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(k_u64_ccc,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

