

#include "Kuiper_GEMM_SHMem.h"

__global__

static void
__hoisted_0(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    float_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

float_t
*Kuiper_GEMM_SHMem_matmul_f32_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_0,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    double_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

double_t
*Kuiper_GEMM_SHMem_matmul_f64_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_1,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    uint32_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_2,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    uint64_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_3,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    float_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

float_t
*Kuiper_GEMM_SHMem_matmul_f32_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_4,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    double_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

double_t
*Kuiper_GEMM_SHMem_matmul_f64_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_5,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    uint32_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_6,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    uint64_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_7,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    float_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile32_rrr(
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
  KPR_KCALL(__hoisted_8,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    double_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile32_rrr(
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
  KPR_KCALL(__hoisted_9,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    uint32_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile32_rrr(
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
  KPR_KCALL(__hoisted_10,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    uint64_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile32_rrr(
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
  KPR_KCALL(__hoisted_11,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    float_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile32_ccc(
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
  KPR_KCALL(__hoisted_12,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    double_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile32_ccc(
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
  KPR_KCALL(__hoisted_13,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    uint32_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile32_ccc(
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
  KPR_KCALL(__hoisted_14,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    uint64_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile32_ccc(
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
  KPR_KCALL(__hoisted_15,
    mrows * mcols,
    (size_t)1024U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    float_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile16_rrr(
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
  KPR_KCALL(__hoisted_16,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    double_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile16_rrr(
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
  KPR_KCALL(__hoisted_17,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    uint32_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile16_rrr(
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
  KPR_KCALL(__hoisted_18,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    uint64_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile16_rrr(
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
  KPR_KCALL(__hoisted_19,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    float_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile16_ccc(
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
  KPR_KCALL(__hoisted_20,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    double_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile16_ccc(
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
  KPR_KCALL(__hoisted_21,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    uint32_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile16_ccc(
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
  KPR_KCALL(__hoisted_22,
    mrows * mcols,
    (size_t)256U,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    uint64_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile16_ccc(
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
  KPR_KCALL(__hoisted_23,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_24(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    float_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f32_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_24,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    double_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_25,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    uint32_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u32_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_26,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    uint64_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_27,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    float_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f32_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_28,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    double_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f64_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_29,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    uint32_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u32_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_30,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    uint64_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u64_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_31,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    float_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile32_rrr(
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
    (size_t)1024U,
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
__hoisted_33(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    double_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile32_rrr(
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
    (size_t)1024U,
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
__hoisted_34(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    uint32_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile32_rrr(
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
    (size_t)1024U,
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
__hoisted_35(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    uint64_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile32_rrr(
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
    (size_t)1024U,
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
__hoisted_36(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    float_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile32_ccc(
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
    (size_t)1024U,
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
__hoisted_37(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    double_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile32_ccc(
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
    (size_t)1024U,
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
__hoisted_38(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    uint32_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile32_ccc(
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
    (size_t)1024U,
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
__hoisted_39(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    uint64_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile32_ccc(
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
    (size_t)1024U,
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
__hoisted_40(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    float_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile16_rrr(
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
    (size_t)256U,
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
__hoisted_41(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    double_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile16_rrr(
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
    (size_t)256U,
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
__hoisted_42(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    uint32_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile16_rrr(
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
    (size_t)256U,
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
__hoisted_43(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    uint64_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile16_rrr(
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
    (size_t)256U,
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
__hoisted_44(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    float_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile16_ccc(
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
    (size_t)256U,
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
__hoisted_45(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    double_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile16_ccc(
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
    (size_t)256U,
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
__hoisted_46(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    uint32_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile16_ccc(
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
    (size_t)256U,
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
__hoisted_47(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    uint64_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile16_ccc(
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
    (size_t)256U,
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
__hoisted_48(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    float_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] =
    beta * gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f32_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_48,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_49(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    double_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] =
    beta * gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f64_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_49,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_50(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    uint32_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] =
    beta * gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u32_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_50,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_51(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    uint64_t v2 = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] =
    beta * gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u64_rrr(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_51,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_52(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    float_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] =
    beta * gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f32_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  float_t *gA4 = gA;
  float_t *gB4 = gB;
  float_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_52,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_53(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    double_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] =
    beta * gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f64_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  double_t *gA4 = gA;
  double_t *gB4 = gB;
  double_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_53,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_54(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    uint32_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] =
    beta * gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u32_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint32_t *gA4 = gA;
  uint32_t *gB4 = gB;
  uint32_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_54,
    mrows * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_55(
  size_t tile,
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / tile;
  size_t bcol = threadIdx.x % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    uint64_t v2 = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    ar[brow * tile + bcol] = v1;
    ar[tile * tile + brow * tile + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[tile * tile + vsk * tile + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] =
    beta * gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u64_ccc(
  size_t tile,
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
  KPR_ASSERT(tile > (size_t)0U);
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mrows = rows / tile;
  size_t mshared = shared / tile;
  size_t mcols = cols / tile;
  uint64_t *gA4 = gA;
  uint64_t *gB4 = gB;
  uint64_t *gC4 = gC;
  KPR_ASSERT(tile > (size_t)0U);
  KPR_KCALL(__hoisted_55,
    mrows * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
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
__hoisted_56(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    float_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] =
    beta * gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile32_rrr(
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
  KPR_KCALL(__hoisted_56,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_57(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    double_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] =
    beta * gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile32_rrr(
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
  KPR_KCALL(__hoisted_57,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_58(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    uint32_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] =
    beta * gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile32_rrr(
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
  KPR_KCALL(__hoisted_58,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_59(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    uint64_t v2 = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] =
    beta * gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile32_rrr(
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
  KPR_KCALL(__hoisted_59,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_60(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    float_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] =
    beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile32_ccc(
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
  KPR_KCALL(__hoisted_60,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_61(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    double_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] =
    beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile32_ccc(
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
  KPR_KCALL(__hoisted_61,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_62(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    uint32_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] =
    beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile32_ccc(
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
  KPR_KCALL(__hoisted_62,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_63(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)32U;
  size_t bcol = threadIdx.x % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    uint64_t v2 = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    ar[brow * (size_t)32U + bcol] = v1;
    ar[(size_t)1024U + brow * (size_t)32U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[(size_t)1024U + vsk * (size_t)32U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] =
    beta * gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile32_ccc(
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
  KPR_KCALL(__hoisted_63,
    mrows * mcols,
    (size_t)1024U,
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
__hoisted_64(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    float_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] =
    beta * gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile16_rrr(
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
  KPR_KCALL(__hoisted_64,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_65(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    double_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] =
    beta * gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile16_rrr(
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
  KPR_KCALL(__hoisted_65,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_66(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    uint32_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] =
    beta * gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile16_rrr(
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
  KPR_KCALL(__hoisted_66,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_67(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    uint64_t v2 = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] =
    beta * gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile16_rrr(
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
  KPR_KCALL(__hoisted_67,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_68(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    float_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    float_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] =
    beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile16_ccc(
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
  KPR_KCALL(__hoisted_68,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_69(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    double_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    double_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] =
    beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile16_ccc(
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
  KPR_KCALL(__hoisted_69,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_70(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint32_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    uint32_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] =
    beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile16_ccc(
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
  KPR_KCALL(__hoisted_70,
    mrows * mcols,
    (size_t)256U,
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
__hoisted_71(
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
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  size_t brow = threadIdx.x / (size_t)16U;
  size_t bcol = threadIdx.x % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    uint64_t v1 = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    uint64_t v2 = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    ar[brow * (size_t)16U + bcol] = v1;
    ar[(size_t)256U + brow * (size_t)16U + bcol] = v2;
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[(size_t)256U + vsk * (size_t)16U + bcol];
      sk = vsk + (size_t)1U;
    }
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] =
    beta * gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] + alpha * sum;
}

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile16_ccc(
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
  KPR_KCALL(__hoisted_71,
    mrows * mcols,
    (size_t)256U,
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

