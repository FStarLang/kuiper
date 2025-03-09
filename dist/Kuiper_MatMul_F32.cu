

#include "Kuiper_MatMul_F32.h"

__global__

void
Kuiper_MatMul_F32_kernel_f32(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t tid = blockIdx_x();
  size_t trow = tid / cols;
  size_t tcol = tid % cols;
  size_t i = (size_t)0U;
  float_t sum = (float_t)0.0f;
  while (i < shared)
  {
    size_t vi = i;
    sum += gA[trow * shared + vi] * gB[vi * cols + tcol];
    i = vi + (size_t)1U;
  }
  gC[trow * cols + tcol] = sum;
}

float_t
*Kuiper_MatMul_F32_matmul_f32(size_t rows, size_t shared, size_t cols, float_t *a, float_t *b)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(Kuiper_MatMul_F32_kernel_f32,
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
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    for (uint32_t _i = 0U; _i < rows * cols; ++_i)
      c[_i] = (float_t)0.0f;
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

