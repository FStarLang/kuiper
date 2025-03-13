

#include "Kuiper_MatMul_Naive_Inst.h"

__global__

static void
__hoisted_0(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t shared0,
  float_t *gA,
  size_t cols1,
  float_t *gB,
  size_t cols2,
  float_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(cols0);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(cols1);
  size_t id = blockIdx_x();
  size_t trow = id / cols2;
  size_t tcol = id % cols2;
  size_t k = (size_t)0U;
  float_t sum = (float_t)0.0f;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[trow * shared0 + vk] * gB[vk * cols2 + tcol];
    k = vk + (size_t)1U;
  }
  gC[trow * cols2 + tcol] = sum;
}

float_t
*Kuiper_MatMul_Naive_Inst_matmul_f32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    shared,
    gA,
    cols,
    gB,
    cols,
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

__global__

static void
__hoisted_1(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t shared0,
  double_t *gA,
  size_t cols1,
  double_t *gB,
  size_t cols2,
  double_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(cols0);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(cols1);
  size_t id = blockIdx_x();
  size_t trow = id / cols2;
  size_t tcol = id % cols2;
  size_t k = (size_t)0U;
  double_t sum = (double_t)0.0l;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[trow * shared0 + vk] * gB[vk * cols2 + tcol];
    k = vk + (size_t)1U;
  }
  gC[trow * cols2 + tcol] = sum;
}

double_t
*Kuiper_MatMul_Naive_Inst_matmul_f64_rrr(
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
  KPR_KCALL(__hoisted_1,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    shared,
    gA,
    cols,
    gB,
    cols,
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

__global__

static void
__hoisted_2(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t shared0,
  uint32_t *gA,
  size_t cols1,
  uint32_t *gB,
  size_t cols2,
  uint32_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(cols0);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(cols1);
  size_t id = blockIdx_x();
  size_t trow = id / cols2;
  size_t tcol = id % cols2;
  size_t k = (size_t)0U;
  uint32_t sum = 0U;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[trow * shared0 + vk] * gB[vk * cols2 + tcol];
    k = vk + (size_t)1U;
  }
  gC[trow * cols2 + tcol] = sum;
}

uint32_t
*Kuiper_MatMul_Naive_Inst_matmul_u32_rrr(
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
  KPR_KCALL(__hoisted_2,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    shared,
    gA,
    cols,
    gB,
    cols,
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
__hoisted_3(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t shared0,
  uint64_t *gA,
  size_t cols1,
  uint64_t *gB,
  size_t cols2,
  uint64_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(cols0);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(cols1);
  size_t id = blockIdx_x();
  size_t trow = id / cols2;
  size_t tcol = id % cols2;
  size_t k = (size_t)0U;
  uint64_t sum = 0ULL;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[trow * shared0 + vk] * gB[vk * cols2 + tcol];
    k = vk + (size_t)1U;
  }
  gC[trow * cols2 + tcol] = sum;
}

uint64_t
*Kuiper_MatMul_Naive_Inst_matmul_u64_rrr(
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
  KPR_KCALL(__hoisted_3,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    shared,
    gA,
    cols,
    gB,
    cols,
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
__hoisted_4(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t rows,
  float_t *gA,
  size_t shared0,
  float_t *gB,
  size_t rows0,
  float_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t id = blockIdx_x();
  size_t trow = id / cols0;
  size_t tcol = id % cols0;
  size_t k = (size_t)0U;
  float_t sum = (float_t)0.0f;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[vk * rows0 + trow] * gB[tcol * shared0 + vk];
    k = vk + (size_t)1U;
  }
  gC[tcol * rows0 + trow] = sum;
}

float_t
*Kuiper_MatMul_Naive_Inst_matmul_f32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U * (rows * shared));
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U * (shared * cols));
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U * (rows * cols));
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_4,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    rows,
    gA,
    shared,
    gB,
    rows,
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

__global__

static void
__hoisted_5(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t rows,
  double_t *gA,
  size_t shared0,
  double_t *gB,
  size_t rows0,
  double_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t id = blockIdx_x();
  size_t trow = id / cols0;
  size_t tcol = id % cols0;
  size_t k = (size_t)0U;
  double_t sum = (double_t)0.0l;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[vk * rows0 + trow] * gB[tcol * shared0 + vk];
    k = vk + (size_t)1U;
  }
  gC[tcol * rows0 + trow] = sum;
}

double_t
*Kuiper_MatMul_Naive_Inst_matmul_f64_ccc(
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
  KPR_KCALL(__hoisted_5,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    rows,
    gA,
    shared,
    gB,
    rows,
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

__global__

static void
__hoisted_6(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t rows,
  uint32_t *gA,
  size_t shared0,
  uint32_t *gB,
  size_t rows0,
  uint32_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t id = blockIdx_x();
  size_t trow = id / cols0;
  size_t tcol = id % cols0;
  size_t k = (size_t)0U;
  uint32_t sum = 0U;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[vk * rows0 + trow] * gB[tcol * shared0 + vk];
    k = vk + (size_t)1U;
  }
  gC[tcol * rows0 + trow] = sum;
}

uint32_t
*Kuiper_MatMul_Naive_Inst_matmul_u32_ccc(
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
  KPR_KCALL(__hoisted_6,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    rows,
    gA,
    shared,
    gB,
    rows,
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
__hoisted_7(
  size_t cols,
  size_t cols0,
  size_t shared,
  size_t rows,
  uint64_t *gA,
  size_t shared0,
  uint64_t *gB,
  size_t rows0,
  uint64_t *gC
)
{
  KRML_MAYBE_UNUSED_VAR(cols);
  KRML_MAYBE_UNUSED_VAR(shared);
  KRML_MAYBE_UNUSED_VAR(rows);
  size_t id = blockIdx_x();
  size_t trow = id / cols0;
  size_t tcol = id % cols0;
  size_t k = (size_t)0U;
  uint64_t sum = 0ULL;
  while (k < shared0)
  {
    size_t vk = k;
    sum += gA[vk * rows0 + trow] * gB[tcol * shared0 + vk];
    k = vk + (size_t)1U;
  }
  gC[tcol * rows0 + trow] = sum;
}

uint64_t
*Kuiper_MatMul_Naive_Inst_matmul_u64_ccc(
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
  KPR_KCALL(__hoisted_7,
    rows * cols,
    (size_t)1U,
    (size_t)4U,
    (size_t)0U,
    cols,
    cols,
    shared,
    rows,
    gA,
    shared,
    gB,
    rows,
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

