

#include "Kuiper_MatMul_F64.h"

__global__

static void
k_f64_rrr(size_t rows, size_t shared, size_t cols, double_t *gA, double_t *gB, double_t *gC)
{
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  double_t sum = (double_t)0.0l;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[trow * shared + vi] * gB[vi * cols + tcol];
    i = vi + (size_t)1U;
  }
  gC[trow * cols + tcol] = sum;
}

double_t
*Kuiper_MatMul_F64_matmul_f64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U * (rows * shared));
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U * (shared * cols));
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(k_f64_rrr,
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
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    for (uint32_t _i = 0U; _i < rows * cols; ++_i)
      c[_i] = (double_t)0.0l;
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

