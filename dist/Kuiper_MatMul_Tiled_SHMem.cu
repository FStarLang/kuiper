

#include "Kuiper_MatMul_Tiled_SHMem.h"

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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    ar[tid + tile * tile] = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

float_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f32_rrr(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_0,
    rows / tile * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
    shared,
    cols,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    ar[tid + tile * tile] = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

double_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f64_rrr(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_1,
    rows / tile * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
    shared,
    cols,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    ar[tid + tile * tile] = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u32_rrr(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_2,
    rows / tile * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
    shared,
    cols,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * tile + brow) * shared + vbk * tile + bcol];
    ar[tid + tile * tile] = gB4[(vbk * tile + brow) * cols + mcol * tile + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * tile + brow) * cols + mcol * tile + bcol] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u64_rrr(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_3,
    rows / tile * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
    shared,
    cols,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    ar[tid + tile * tile] = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

float_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f32_ccc(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_4,
    rows / tile * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
    rows,
    shared,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    ar[tid + tile * tile] = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

double_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f64_ccc(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_5,
    rows / tile * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
    rows,
    shared,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    ar[tid + tile * tile] = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u32_ccc(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_6,
    rows / tile * mcols,
    tile * tile,
    (size_t)4U,
    (size_t)2U * tile * tile,
    tile,
    rows,
    shared,
    shared / tile,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / tile;
  size_t bcol = tid % tile;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * tile + bcol) * rows + mrow * tile + brow];
    ar[tid + tile * tile] = gB4[(mcol * tile + bcol) * shared + vbk * tile + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < tile)
    {
      size_t vsk = sk;
      sum += ar[brow * tile + vsk] * ar[vsk * tile + bcol + tile * tile];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * tile + bcol) * rows + mrow * tile + brow] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u64_ccc(
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
  KPR_GUARD(rows % tile == (size_t)0U);
  KPR_GUARD(shared % tile == (size_t)0U);
  KPR_GUARD(cols % tile == (size_t)0U);
  size_t mcols = cols / tile;
  KPR_KCALL(__hoisted_7,
    rows / tile * mcols,
    tile * tile,
    (size_t)8U,
    (size_t)2U * tile * tile,
    tile,
    rows,
    shared,
    shared / tile,
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
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    ar[tid + (size_t)1024U] = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

float_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f32_tile32_rrr(
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
    (size_t)4U,
    (size_t)2048U,
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
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    ar[tid + (size_t)1024U] = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

double_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f64_tile32_rrr(
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
    (size_t)8U,
    (size_t)2048U,
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
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    ar[tid + (size_t)1024U] = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u32_tile32_rrr(
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
    (size_t)4U,
    (size_t)2048U,
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
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)32U + brow) * shared + vbk * (size_t)32U + bcol];
    ar[tid + (size_t)1024U] = gB4[(vbk * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)32U + brow) * cols + mcol * (size_t)32U + bcol] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u64_tile32_rrr(
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
    (size_t)8U,
    (size_t)2048U,
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
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    ar[tid + (size_t)1024U] = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

float_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f32_tile32_ccc(
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
    (size_t)4U,
    (size_t)2048U,
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
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    ar[tid + (size_t)1024U] = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

double_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f64_tile32_ccc(
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
    (size_t)8U,
    (size_t)2048U,
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
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    ar[tid + (size_t)1024U] = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u32_tile32_ccc(
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
    (size_t)4U,
    (size_t)2048U,
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
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)32U;
  size_t bcol = tid % (size_t)32U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow];
    ar[tid + (size_t)1024U] = gB4[(mcol * (size_t)32U + bcol) * shared + vbk * (size_t)32U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)32U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)32U + vsk] * ar[vsk * (size_t)32U + bcol + (size_t)1024U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)32U + bcol) * rows + mrow * (size_t)32U + brow] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u64_tile32_ccc(
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
    (size_t)8U,
    (size_t)2048U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    ar[tid + (size_t)256U] = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

float_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f32_tile16_rrr(
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
  KPR_KCALL(__hoisted_16,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)4U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    ar[tid + (size_t)256U] = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

double_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f64_tile16_rrr(
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
  KPR_KCALL(__hoisted_17,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)8U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    ar[tid + (size_t)256U] = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u32_tile16_rrr(
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
  KPR_KCALL(__hoisted_18,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)4U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(mrow * (size_t)16U + brow) * shared + vbk * (size_t)16U + bcol];
    ar[tid + (size_t)256U] = gB4[(vbk * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mrow * (size_t)16U + brow) * cols + mcol * (size_t)16U + bcol] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u64_tile16_rrr(
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
  KPR_KCALL(__hoisted_19,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)8U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  float_t *ar = (float_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  float_t sum = (float_t)0.0f;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    ar[tid + (size_t)256U] = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

float_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f32_tile16_ccc(
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
  KPR_KCALL(__hoisted_20,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)4U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  double_t *ar = (double_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  double_t sum = (double_t)0.0l;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    ar[tid + (size_t)256U] = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

double_t
*Kuiper_MatMul_Tiled_SHMem_matmul_f64_tile16_ccc(
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
  KPR_KCALL(__hoisted_21,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)8U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint32_t *ar = (uint32_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  uint32_t sum = 0U;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    ar[tid + (size_t)256U] = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

uint32_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u32_tile16_ccc(
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
  KPR_KCALL(__hoisted_22,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)4U,
    (size_t)512U,
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
  size_t bid = blockIdx_x();
  size_t tid = threadIdx_x();
  uint64_t *ar = (uint64_t *)KPR_SHMEM();
  size_t mrow = bid / mcols;
  size_t mcol = bid % mcols;
  size_t brow = tid / (size_t)16U;
  size_t bcol = tid % (size_t)16U;
  uint64_t sum = 0ULL;
  size_t bk = (size_t)0U;
  while (bk < mshared)
  {
    size_t vbk = bk;
    ar[tid] = gA4[(vbk * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow];
    ar[tid + (size_t)256U] = gB4[(mcol * (size_t)16U + bcol) * shared + vbk * (size_t)16U + brow];
    __syncthreads();
    size_t sk = (size_t)0U;
    while (sk < (size_t)16U)
    {
      size_t vsk = sk;
      sum += ar[brow * (size_t)16U + vsk] * ar[vsk * (size_t)16U + bcol + (size_t)256U];
      sk = vsk + (size_t)1U;
    }
    __syncthreads();
    bk = vbk + (size_t)1U;
  }
  gC4[(mcol * (size_t)16U + bcol) * rows + mrow * (size_t)16U + brow] = sum;
}

uint64_t
*Kuiper_MatMul_Tiled_SHMem_matmul_u64_tile16_ccc(
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
  KPR_KCALL(__hoisted_23,
    rows / (size_t)16U * mcols,
    (size_t)256U,
    (size_t)8U,
    (size_t)512U,
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

