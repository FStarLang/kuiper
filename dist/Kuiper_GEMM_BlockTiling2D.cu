

#include "Kuiper_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting matmul_f32_64x64x8_8x8_rrr_rr
*/
static void __hoisted_0(size_t shared, size_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < shared / (size_t)8U)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    float_t *tileA = gA;
    size_t i1 = threadIdx.x;
    for (; i1 < (size_t)512U; i1 += (size_t)64U)
      sA[i1 / (size_t)8U * (size_t)8U + i1 % (size_t)8U] =
        tileA[(mrow * (size_t)64U + i1 / (size_t)8U) * shared +
          __anf01 * (size_t)8U + i1 % (size_t)8U];
    float_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)512U; i += (size_t)64U)
      sB[i / (size_t)64U * (size_t)64U + i % (size_t)64U] =
        tileB[(__anf01 * (size_t)8U + i / (size_t)64U) * cols + mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      for (; i0 < (size_t)8U; i0 += (size_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (size_t)8U * (size_t)8U + i0) * (size_t)8U + dotIdx];
      size_t i1 = (size_t)0U;
      for (; i1 < (size_t)8U; i1 += (size_t)1U)
        rBrow[i1] = sB[dotIdx * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + i1];
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  float_t *t_tile = gC;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
      t_tile[(blockIdx.x / (cols / (size_t)64U) * (size_t)64U +
        threadIdx.x / (size_t)8U * (size_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (size_t)64U) * (size_t)64U +
          threadIdx.x % (size_t)8U * (size_t)8U + resIdxN]
      = rchProd[resIdxM * (size_t)8U + resIdxN];
    resIdxM += (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_64x64x8_8x8_rrr_rr(
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
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)8U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_0,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)64U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_32x32x32_32x8_rrr_rr
*/
static void __hoisted_1(size_t shared, size_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)4096U);
  size_t num_n_tiles = cols / (size_t)32U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[32U];
  memset(rAcol, 0U, (size_t)32U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[256U];
  memset(rchProd, 0U, (size_t)256U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < shared / (size_t)32U)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    float_t *tileA = gA;
    size_t i1 = threadIdx.x;
    for (; i1 < (size_t)1024U; i1 += (size_t)4U)
      sA[i1 / (size_t)32U * (size_t)32U + i1 % (size_t)32U] =
        tileA[(mrow * (size_t)32U + i1 / (size_t)32U) * shared +
          __anf01 * (size_t)32U + i1 % (size_t)32U];
    float_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)4U)
      sB[i / (size_t)32U * (size_t)32U + i % (size_t)32U] =
        tileB[(__anf01 * (size_t)32U + i / (size_t)32U) * cols +
          mcol * (size_t)32U + i % (size_t)32U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)32U)
    {
      size_t i0 = (size_t)0U;
      for (; i0 < (size_t)32U; i0 += (size_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (size_t)4U * (size_t)32U + i0) * (size_t)32U + dotIdx];
      size_t i1 = (size_t)0U;
      for (; i1 < (size_t)8U; i1 += (size_t)1U)
        rBrow[i1] = sB[dotIdx * (size_t)32U + threadIdx.x % (size_t)4U * (size_t)8U + i1];
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)32U)
      {
        size_t resIdxN = (size_t)0U;
        for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  float_t *t_tile = gC;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)32U)
  {
    size_t resIdxN = (size_t)0U;
    for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
      t_tile[(blockIdx.x / (cols / (size_t)32U) * (size_t)32U +
        threadIdx.x / (size_t)4U * (size_t)32U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (size_t)32U) * (size_t)32U +
          threadIdx.x % (size_t)4U * (size_t)8U + resIdxN]
      = rchProd[resIdxM * (size_t)8U + resIdxN];
    resIdxM += (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_32x32x32_32x8_rrr_rr(
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
  KPR_KCALL(__hoisted_1,
    rows / (size_t)32U * (cols / (size_t)32U),
    (size_t)4U,
    (size_t)8192U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x8_8x8_rrr_rr
*/
static void
__hoisted_2(
  float_t alpha,
  float_t beta,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < shared / (size_t)8U)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    float_t *tileA = gA;
    size_t i1 = threadIdx.x;
    for (; i1 < (size_t)512U; i1 += (size_t)64U)
      sA[i1 / (size_t)8U * (size_t)8U + i1 % (size_t)8U] =
        tileA[(mrow * (size_t)64U + i1 / (size_t)8U) * shared +
          __anf01 * (size_t)8U + i1 % (size_t)8U];
    float_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)512U; i += (size_t)64U)
      sB[i / (size_t)64U * (size_t)64U + i % (size_t)64U] =
        tileB[(__anf01 * (size_t)8U + i / (size_t)64U) * cols + mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      for (; i0 < (size_t)8U; i0 += (size_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (size_t)8U * (size_t)8U + i0) * (size_t)8U + dotIdx];
      size_t i1 = (size_t)0U;
      for (; i1 < (size_t)8U; i1 += (size_t)1U)
        rBrow[i1] = sB[dotIdx * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + i1];
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  float_t *t_tile = gC;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
      t_tile[(blockIdx.x / (cols / (size_t)64U) * (size_t)64U +
        threadIdx.x / (size_t)8U * (size_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (size_t)64U) * (size_t)64U +
          threadIdx.x % (size_t)8U * (size_t)8U + resIdxN]
      =
        beta *
          t_tile[(blockIdx.x / (cols / (size_t)64U) * (size_t)64U +
            threadIdx.x / (size_t)8U * (size_t)8U + resIdxM)
          * cols
          +
            blockIdx.x % (cols / (size_t)64U) * (size_t)64U +
              threadIdx.x % (size_t)8U * (size_t)8U + resIdxN]
        + alpha * rchProd[resIdxM * (size_t)8U + resIdxN];
    resIdxM += (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rrr_rr(
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
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)8U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_2,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)64U,
    (size_t)4096U,
    alpha,
    beta,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x8_8x8_rrr_rr
*/
static void
__hoisted_3(
  float_t alpha,
  float_t beta,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)4096U);
  size_t num_n_tiles = cols / (size_t)128U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < shared / (size_t)8U)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    float_t *tileA = gA;
    size_t i1 = threadIdx.x;
    for (; i1 < (size_t)1024U; i1 += (size_t)256U)
      sA[i1 / (size_t)8U * (size_t)8U + i1 % (size_t)8U] =
        tileA[(mrow * (size_t)128U + i1 / (size_t)8U) * shared +
          __anf01 * (size_t)8U + i1 % (size_t)8U];
    float_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)256U)
      sB[i / (size_t)128U * (size_t)128U + i % (size_t)128U] =
        tileB[(__anf01 * (size_t)8U + i / (size_t)128U) * cols +
          mcol * (size_t)128U + i % (size_t)128U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      for (; i0 < (size_t)8U; i0 += (size_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (size_t)16U * (size_t)8U + i0) * (size_t)8U + dotIdx];
      size_t i1 = (size_t)0U;
      for (; i1 < (size_t)8U; i1 += (size_t)1U)
        rBrow[i1] = sB[dotIdx * (size_t)128U + threadIdx.x % (size_t)16U * (size_t)8U + i1];
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  float_t *t_tile = gC;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
      t_tile[(blockIdx.x / (cols / (size_t)128U) * (size_t)128U +
        threadIdx.x / (size_t)16U * (size_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (size_t)128U) * (size_t)128U +
          threadIdx.x % (size_t)16U * (size_t)8U + resIdxN]
      =
        beta *
          t_tile[(blockIdx.x / (cols / (size_t)128U) * (size_t)128U +
            threadIdx.x / (size_t)16U * (size_t)8U + resIdxM)
          * cols
          +
            blockIdx.x % (cols / (size_t)128U) * (size_t)128U +
              threadIdx.x % (size_t)16U * (size_t)8U + resIdxN]
        + alpha * rchProd[resIdxM * (size_t)8U + resIdxN];
    resIdxM += (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr_rr(
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
  KPR_GUARD(rows % (size_t)128U == (size_t)0U);
  KPR_GUARD(shared % (size_t)8U == (size_t)0U);
  KPR_GUARD(cols % (size_t)128U == (size_t)0U);
  KPR_KCALL(__hoisted_3,
    rows / (size_t)128U * (cols / (size_t)128U),
    (size_t)256U,
    (size_t)8192U,
    alpha,
    beta,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x8_8x8_rrr_cr
*/
static void
__hoisted_4(
  float_t alpha,
  float_t beta,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)4096U);
  size_t num_n_tiles = cols / (size_t)128U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < shared / (size_t)8U)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    float_t *tileA = gA;
    size_t i1 = threadIdx.x;
    for (; i1 < (size_t)1024U; i1 += (size_t)256U)
      sA[i1 % (size_t)8U * (size_t)128U + i1 / (size_t)8U] =
        tileA[(mrow * (size_t)128U + i1 / (size_t)8U) * shared +
          __anf01 * (size_t)8U + i1 % (size_t)8U];
    float_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)256U)
      sB[i / (size_t)128U * (size_t)128U + i % (size_t)128U] =
        tileB[(__anf01 * (size_t)8U + i / (size_t)128U) * cols +
          mcol * (size_t)128U + i % (size_t)128U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      for (; i0 < (size_t)8U; i0 += (size_t)1U)
        rAcol[i0] = sA[dotIdx * (size_t)128U + threadIdx.x / (size_t)16U * (size_t)8U + i0];
      size_t i1 = (size_t)0U;
      for (; i1 < (size_t)8U; i1 += (size_t)1U)
        rBrow[i1] = sB[dotIdx * (size_t)128U + threadIdx.x % (size_t)16U * (size_t)8U + i1];
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  float_t *t_tile = gC;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    for (; resIdxN < (size_t)8U; resIdxN += (size_t)1U)
      t_tile[(blockIdx.x / (cols / (size_t)128U) * (size_t)128U +
        threadIdx.x / (size_t)16U * (size_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (size_t)128U) * (size_t)128U +
          threadIdx.x % (size_t)16U * (size_t)8U + resIdxN]
      =
        beta *
          t_tile[(blockIdx.x / (cols / (size_t)128U) * (size_t)128U +
            threadIdx.x / (size_t)16U * (size_t)8U + resIdxM)
          * cols
          +
            blockIdx.x % (cols / (size_t)128U) * (size_t)128U +
              threadIdx.x % (size_t)16U * (size_t)8U + resIdxN]
        + alpha * rchProd[resIdxM * (size_t)8U + resIdxN];
    resIdxM += (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr_cr(
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
  KPR_GUARD(rows % (size_t)128U == (size_t)0U);
  KPR_GUARD(shared % (size_t)8U == (size_t)0U);
  KPR_GUARD(cols % (size_t)128U == (size_t)0U);
  KPR_KCALL(__hoisted_4,
    rows / (size_t)128U * (cols / (size_t)128U),
    (size_t)256U,
    (size_t)8192U,
    alpha,
    beta,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

