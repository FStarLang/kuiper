

#include "Kuiper_GEMM_BlockTiling1D.h"

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
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      float_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile32_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_0,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      double_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

double_t
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile32_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_1,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      uint32_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint32_t
*Kuiper_GEMM_BlockTiling1D_matmul_u32_tile32_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_2,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      uint64_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint64_t
*Kuiper_GEMM_BlockTiling1D_matmul_u64_tile32_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_3,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      float_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile32_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_4,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      double_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

double_t
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile32_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_5,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      uint32_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint32_t
*Kuiper_GEMM_BlockTiling1D_matmul_u32_tile32_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_6,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      uint64_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint64_t
*Kuiper_GEMM_BlockTiling1D_matmul_u64_tile32_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_7,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      float_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile16_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_8,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      double_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

double_t
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile16_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_9,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      uint32_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint32_t
*Kuiper_GEMM_BlockTiling1D_matmul_u32_tile16_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_10,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      uint64_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint64_t
*Kuiper_GEMM_BlockTiling1D_matmul_u64_tile16_rrr(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_11,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      float_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile16_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_12,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      double_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

double_t
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile16_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_13,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      uint32_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint32_t
*Kuiper_GEMM_BlockTiling1D_matmul_u32_tile16_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_14,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      uint64_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

uint64_t
*Kuiper_GEMM_BlockTiling1D_matmul_u64_tile16_ccc(
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
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_15,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
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
__hoisted_16(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      float_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_16,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_17(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      double_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_17,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_18(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      uint32_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_18,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_19(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      uint64_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_19,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_20(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      float_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_20,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_21(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      double_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_21,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_22(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      uint32_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_22,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_23(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      uint64_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_23,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_24(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      float_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_24,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_25(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      double_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_25,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_26(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      uint32_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_26,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_27(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      uint64_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_27,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_28(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      float_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_28,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_29(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      double_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_29,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_30(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      uint32_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_30,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_31(
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      uint64_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] = sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_31,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_32(
  float_t alpha,
  float_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      float_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] =
      beta * gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_32,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_33(
  double_t alpha,
  double_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      double_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] =
      beta * gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_33,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_34(
  uint32_t alpha,
  uint32_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      uint32_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] =
      beta * gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_34,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_35(
  uint64_t alpha,
  uint64_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(mrow * (size_t)32U + vi) * shared + vbk * (size_t)32U + threadIdx_x()];
      uint64_t v2 = gB4[(vbk * (size_t)32U + vi) * cols + mcol * (size_t)32U + threadIdx_x()];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] =
      beta * gC4[(mrow * (size_t)32U + vrow) * cols + mcol * (size_t)32U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_35,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_36(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      float_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] =
      beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_ccc(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_36,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_37(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[32U];
  for (uint32_t _i = 0U; _i < (size_t)32U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      double_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] =
      beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_ccc(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_37,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_38(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      uint32_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] =
      beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_ccc(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_38,
    mrows * mcols,
    (size_t)32U,
    (size_t)4U,
    (size_t)2048U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_39(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[32U];
  memset(sums, 0U, (size_t)32U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)32U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(vbk * (size_t)32U + threadIdx_x()) * rows + mrow * (size_t)32U + vi];
      uint64_t v2 = gB4[(mcol * (size_t)32U + threadIdx_x()) * shared + vbk * (size_t)32U + vi];
      ar[vi * (size_t)32U + threadIdx_x()] = v1;
      ar[(size_t)1024U + vi * (size_t)32U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      while (i < (size_t)32U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)32U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)32U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] =
      beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_ccc(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  size_t mrows = rows / (size_t)32U;
  size_t mshared = shared / (size_t)32U;
  size_t mcols = cols / (size_t)32U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_39,
    mrows * mcols,
    (size_t)32U,
    (size_t)8U,
    (size_t)2048U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_40(
  float_t alpha,
  float_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      float_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] =
      beta * gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_40,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_41(
  double_t alpha,
  double_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      double_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] =
      beta * gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_41,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_42(
  uint32_t alpha,
  uint32_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      uint32_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] =
      beta * gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_42,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_43(
  uint64_t alpha,
  uint64_t beta,
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(mrow * (size_t)16U + vi) * shared + vbk * (size_t)16U + threadIdx_x()];
      uint64_t v2 = gB4[(vbk * (size_t)16U + vi) * cols + mcol * (size_t)16U + threadIdx_x()];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] =
      beta * gC4[(mrow * (size_t)16U + vrow) * cols + mcol * (size_t)16U + bcol] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_43,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    alpha,
    beta,
    shared,
    cols,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_44(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  float_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      float_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      float_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      float_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] =
      beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_ccc(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_44,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_45(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  double_t *gA4,
  double_t *gB4,
  double_t *gC4
)
{
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  double_t sums[16U];
  for (uint32_t _i = 0U; _i < (size_t)16U; ++_i)
    sums[_i] = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      double_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      double_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      double_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] =
      beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_ccc(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_45,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_46(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint32_t *gA4,
  uint32_t *gB4,
  uint32_t *gC4
)
{
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint32_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint32_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint32_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      uint32_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint32_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] =
      beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_ccc(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_46,
    mrows * mcols,
    (size_t)16U,
    (size_t)4U,
    (size_t)512U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

__global__

static void
__hoisted_47(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t mshared,
  size_t mcols,
  uint64_t *gA4,
  uint64_t *gB4,
  uint64_t *gC4
)
{
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = blockIdx_x() / mcols;
  size_t mcol = blockIdx_x() % mcols;
  size_t bcol = threadIdx_x();
  uint64_t sums[16U];
  memset(sums, 0U, (size_t)16U * sizeof (uint64_t));
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    __syncthreads();
    size_t i0 = (size_t)0U;
    while (i0 < (size_t)16U)
    {
      size_t vi = i0;
      uint64_t v1 = gA4[(vbk * (size_t)16U + threadIdx_x()) * rows + mrow * (size_t)16U + vi];
      uint64_t v2 = gB4[(mcol * (size_t)16U + threadIdx_x()) * shared + vbk * (size_t)16U + vi];
      ar[vi * (size_t)16U + threadIdx_x()] = v1;
      ar[(size_t)256U + vi * (size_t)16U + threadIdx_x()] = v2;
      i0 = vi + (size_t)1U;
    }
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      size_t i = (size_t)0U;
      uint64_t v2 = ar[(size_t)256U + vsk * (size_t)16U + bcol];
      while (i < (size_t)16U)
      {
        size_t vi = i;
        sums[vi] += ar[vi * (size_t)16U + vsk] * v2;
        i = vi + (size_t)1U;
      }
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  size_t row = (size_t)0U;
  while (row < (size_t)16U)
  {
    size_t vrow = row;
    gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] =
      beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + vrow] +
        alpha * sums[vrow];
    row = vrow + (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_ccc(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(true);
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  size_t mrows = rows / (size_t)16U;
  size_t mshared = shared / (size_t)16U;
  size_t mcols = cols / (size_t)16U;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(true);
  KPR_KCALL(__hoisted_47,
    mrows * mcols,
    (size_t)16U,
    (size_t)8U,
    (size_t)512U,
    alpha,
    beta,
    rows,
    shared,
    mshared,
    mcols,
    gA4,
    gB4,
    gC4);
  cudaDeviceSynchronize();
}

