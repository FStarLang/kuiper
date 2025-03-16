

#include "Kuiper_MatMul_Tiled.h"

__global__

static void
__hoisted_0(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t sum1 = (float_t)0.0f;
    size_t k = (size_t)0U;
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + vk] *
          gB4[(vbk * (size_t)16U + vk) * cols + mcol * (size_t)16U + bcol];
      k = vk + (size_t)1U;
    }
    float_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)16U + i) * cols + bj * (size_t)16U + j] = sum;
}

float_t
*Kuiper_MatMul_Tiled_matmul_f32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_0,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)16U,
    mcols,
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

__global__

static void
__hoisted_1(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t sum1 = (double_t)0.0l;
    size_t k = (size_t)0U;
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + vk] *
          gB4[(vbk * (size_t)16U + vk) * cols + mcol * (size_t)16U + bcol];
      k = vk + (size_t)1U;
    }
    double_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)16U + i) * cols + bj * (size_t)16U + j] = sum;
}

double_t
*Kuiper_MatMul_Tiled_matmul_f64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_1,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)16U,
    mcols,
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

__global__

static void
__hoisted_2(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t sum1 = 0U;
    size_t k = (size_t)0U;
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + vk] *
          gB4[(vbk * (size_t)16U + vk) * cols + mcol * (size_t)16U + bcol];
      k = vk + (size_t)1U;
    }
    uint32_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)16U + i) * cols + bj * (size_t)16U + j] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_matmul_u32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_2,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)16U,
    mcols,
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
__hoisted_3(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
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
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + vk] *
          gB4[(vbk * (size_t)16U + vk) * cols + mcol * (size_t)16U + bcol];
      k = vk + (size_t)1U;
    }
    uint64_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)16U + i) * cols + bj * (size_t)16U + j] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_3,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)16U,
    mcols,
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
__hoisted_4(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t sum1 = (float_t)0.0f;
    size_t k = (size_t)0U;
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)16U + vk) * rows + mrow * (size_t)16U + brow] *
          gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + vk];
      k = vk + (size_t)1U;
    }
    float_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)16U + j) * rows + bi * (size_t)16U + i] = sum;
}

float_t
*Kuiper_MatMul_Tiled_matmul_f32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_4,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)16U,
    mcols,
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

__global__

static void
__hoisted_5(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t sum1 = (double_t)0.0l;
    size_t k = (size_t)0U;
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)16U + vk) * rows + mrow * (size_t)16U + brow] *
          gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + vk];
      k = vk + (size_t)1U;
    }
    double_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)16U + j) * rows + bi * (size_t)16U + i] = sum;
}

double_t
*Kuiper_MatMul_Tiled_matmul_f64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_5,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)16U,
    mcols,
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

__global__

static void
__hoisted_6(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  size_t bi = mrow;
  size_t bj = mcol;
  size_t i = brow;
  size_t j = bcol;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t sum1 = 0U;
    size_t k = (size_t)0U;
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)16U + vk) * rows + mrow * (size_t)16U + brow] *
          gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + vk];
      k = vk + (size_t)1U;
    }
    uint32_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)16U + j) * rows + bi * (size_t)16U + i] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_matmul_u32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_6,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)16U,
    mcols,
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
__hoisted_7(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
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
    while (k < (size_t)16U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)16U + vk) * rows + mrow * (size_t)16U + brow] *
          gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + vk];
      k = vk + (size_t)1U;
    }
    uint64_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)16U + j) * rows + bi * (size_t)16U + i] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mcols = cols / (size_t)16U;
  KPR_KCALL(__hoisted_7,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)16U,
    mcols,
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
__hoisted_8(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
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
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t sum1 = (float_t)0.0f;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + vk] *
          gB4[(vbk * (size_t)32U + vk) * cols + mcol * (size_t)32U + bcol];
      k = vk + (size_t)1U;
    }
    float_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)32U + i) * cols + bj * (size_t)32U + j] = sum;
}

float_t
*Kuiper_MatMul_Tiled_matmul_f32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_8,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)32U,
    mcols,
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

__global__

static void
__hoisted_9(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
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
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t sum1 = (double_t)0.0l;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + vk] *
          gB4[(vbk * (size_t)32U + vk) * cols + mcol * (size_t)32U + bcol];
      k = vk + (size_t)1U;
    }
    double_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)32U + i) * cols + bj * (size_t)32U + j] = sum;
}

double_t
*Kuiper_MatMul_Tiled_matmul_f64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_9,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)32U,
    mcols,
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

__global__

static void
__hoisted_10(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
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
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t sum1 = 0U;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + vk] *
          gB4[(vbk * (size_t)32U + vk) * cols + mcol * (size_t)32U + bcol];
      k = vk + (size_t)1U;
    }
    uint32_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)32U + i) * cols + bj * (size_t)32U + j] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_matmul_u32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_10,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)32U,
    mcols,
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
__hoisted_11(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
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
        gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + vk] *
          gB4[(vbk * (size_t)32U + vk) * cols + mcol * (size_t)32U + bcol];
      k = vk + (size_t)1U;
    }
    uint64_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bi * (size_t)32U + i) * cols + bj * (size_t)32U + j] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_11,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    shared,
    cols,
    shared / (size_t)32U,
    mcols,
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
__hoisted_12(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
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
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t sum1 = (float_t)0.0f;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)32U + vk) * rows + mrow * (size_t)32U + brow] *
          gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + vk];
      k = vk + (size_t)1U;
    }
    float_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)32U + j) * rows + bi * (size_t)32U + i] = sum;
}

float_t
*Kuiper_MatMul_Tiled_matmul_f32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_12,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)32U,
    mcols,
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

__global__

static void
__hoisted_13(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
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
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t sum1 = (double_t)0.0l;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)32U + vk) * rows + mrow * (size_t)32U + brow] *
          gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + vk];
      k = vk + (size_t)1U;
    }
    double_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)32U + j) * rows + bi * (size_t)32U + i] = sum;
}

double_t
*Kuiper_MatMul_Tiled_matmul_f64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_13,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)32U,
    mcols,
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

__global__

static void
__hoisted_14(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
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
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t sum1 = 0U;
    size_t k = (size_t)0U;
    while (k < (size_t)32U)
    {
      size_t vk = k;
      sum1 +=
        gA4[(vbk * (size_t)32U + vk) * rows + mrow * (size_t)32U + brow] *
          gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + vk];
      k = vk + (size_t)1U;
    }
    uint32_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)32U + j) * rows + bi * (size_t)32U + i] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_matmul_u32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_14,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)32U,
    mcols,
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
__hoisted_15(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
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
        gA4[(vbk * (size_t)32U + vk) * rows + mrow * (size_t)32U + brow] *
          gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + vk];
      k = vk + (size_t)1U;
    }
    uint64_t sub = sum1;
    sum += sub;
    bk = vbk + (size_t)1U;
  }
  gC4[(bj * (size_t)32U + j) * rows + bi * (size_t)32U + i] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mcols = cols / (size_t)32U;
  KPR_KCALL(__hoisted_15,
    rows / (size_t)32U * mcols,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    shared / (size_t)32U,
    mcols,
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

