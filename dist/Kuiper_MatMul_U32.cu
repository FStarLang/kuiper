

#include "Kuiper_MatMul_U32.h"

__global__

static void
kernel_u32(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  uint32_t sum = 0U;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[trow * shared + vi] * gB[vi * cols + tcol];
    i = vi + (size_t)1U;
  }
  gC[trow * cols + tcol] = sum;
}

uint32_t
*Kuiper_MatMul_U32_matmul_u32(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(kernel_u32,
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
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__

static void
kernel_u32_(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  uint32_t sum = 0U;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[vi * rows + trow] * gB[tcol * shared + vi];
    i = vi + (size_t)1U;
  }
  gC[tcol * rows + trow] = sum;
}

uint32_t
*Kuiper_MatMul_U32_matmul_u32_(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(kernel_u32_,
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
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__

static void
kernel_u32__(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  uint32_t sum = 0U;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[vi * rows + trow] * gB[tcol * shared + vi];
    i = vi + (size_t)1U;
  }
  gC[trow * cols + tcol] = sum;
}

uint32_t
*Kuiper_MatMul_U32_matmul_u32__(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(kernel_u32__,
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
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__

static void
kernel_u32_mirror(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  uint32_t sum = 0U;
  while (i < shared)
  {
    size_t vi = i;
    sum +=
      gA[rows
      * shared
      - (size_t)1U
      - trow * shared
      - vi]
      * gB[shared * cols - (size_t)1U - vi * cols - tcol];
    i = vi + (size_t)1U;
  }
  gC[rows * cols - (size_t)1U - trow * cols - tcol] = sum;
}

uint32_t
*Kuiper_MatMul_U32_matmul_u32_mirror(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(kernel_u32_mirror,
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
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

