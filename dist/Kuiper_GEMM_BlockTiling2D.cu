

#include "Kuiper_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting matmul_f32_64x64x8_8x8_rrr_rr
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (uint32_t)64U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < shared / (uint32_t)8U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    float_t *tileA = gA;
    uint32_t i1 = threadIdx.x;
    for (; i1 < (uint32_t)512U; i1 += (uint32_t)64U)
      sA[i1] =
        tileA[(mrow * (uint32_t)64U + i1 / (uint32_t)8U) * shared +
          __anf01 * (uint32_t)8U + i1 % (uint32_t)8U];
    float_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)512U; i += (uint32_t)64U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)8U + i / (uint32_t)64U) * cols +
          mcol * (uint32_t)64U + i % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)8U)
    {
      uint32_t i0 = (uint32_t)0U;
      for (; i0 < (uint32_t)8U; i0 += (uint32_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (uint32_t)8U * (uint32_t)8U + i0) * (uint32_t)8U + dotIdx];
      uint32_t i1 = (uint32_t)0U;
      for (; i1 < (uint32_t)8U; i1 += (uint32_t)1U)
        rBrow[i1] = sB[dotIdx * (uint32_t)64U + threadIdx.x % (uint32_t)8U * (uint32_t)8U + i1];
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)8U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
          rchProd[resIdxM * (uint32_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  float_t *t_tile = gC;
  uint32_t resIdxM = (uint32_t)0U;
  while (resIdxM < (uint32_t)8U)
  {
    uint32_t resIdxN = (uint32_t)0U;
    for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
      t_tile[(blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U +
        threadIdx.x / (uint32_t)8U * (uint32_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U +
          threadIdx.x % (uint32_t)8U * (uint32_t)8U + resIdxN]
      = rchProd[resIdxM * (uint32_t)8U + resIdxN];
    resIdxM += (uint32_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_64x64x8_8x8_rrr_rr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)8U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_0,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)64U,
    (uint32_t)4096U,
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
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_32x32x32_32x8_rrr_rr
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)4096U);
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[32U];
  memset(rAcol, 0U, (uint32_t)32U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rchProd[256U];
  memset(rchProd, 0U, (uint32_t)256U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < shared / (uint32_t)32U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    float_t *tileA = gA;
    uint32_t i1 = threadIdx.x;
    for (; i1 < (uint32_t)1024U; i1 += (uint32_t)4U)
      sA[i1] =
        tileA[(mrow * (uint32_t)32U + i1 / (uint32_t)32U) * shared +
          __anf01 * (uint32_t)32U + i1 % (uint32_t)32U];
    float_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)4U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)32U + i / (uint32_t)32U) * cols +
          mcol * (uint32_t)32U + i % (uint32_t)32U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)32U)
    {
      uint32_t i0 = (uint32_t)0U;
      for (; i0 < (uint32_t)32U; i0 += (uint32_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (uint32_t)4U * (uint32_t)32U + i0) * (uint32_t)32U + dotIdx];
      uint32_t i1 = (uint32_t)0U;
      for (; i1 < (uint32_t)8U; i1 += (uint32_t)1U)
        rBrow[i1] = sB[dotIdx * (uint32_t)32U + threadIdx.x % (uint32_t)4U * (uint32_t)8U + i1];
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)32U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
          rchProd[resIdxM * (uint32_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  float_t *t_tile = gC;
  uint32_t resIdxM = (uint32_t)0U;
  while (resIdxM < (uint32_t)32U)
  {
    uint32_t resIdxN = (uint32_t)0U;
    for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
      t_tile[(blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U +
        threadIdx.x / (uint32_t)4U * (uint32_t)32U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U +
          threadIdx.x % (uint32_t)4U * (uint32_t)8U + resIdxN]
      = rchProd[resIdxM * (uint32_t)8U + resIdxN];
    resIdxM += (uint32_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_32x32x32_32x8_rrr_rr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  KPR_KCALL(__hoisted_1,
    rows / (uint32_t)32U * (cols / (uint32_t)32U),
    (uint32_t)4U,
    (uint32_t)8192U,
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
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
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
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (uint32_t)64U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < shared / (uint32_t)8U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    float_t *tileA = gA;
    uint32_t i1 = threadIdx.x;
    for (; i1 < (uint32_t)512U; i1 += (uint32_t)64U)
      sA[i1] =
        tileA[(mrow * (uint32_t)64U + i1 / (uint32_t)8U) * shared +
          __anf01 * (uint32_t)8U + i1 % (uint32_t)8U];
    float_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)512U; i += (uint32_t)64U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)8U + i / (uint32_t)64U) * cols +
          mcol * (uint32_t)64U + i % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)8U)
    {
      uint32_t i0 = (uint32_t)0U;
      for (; i0 < (uint32_t)8U; i0 += (uint32_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (uint32_t)8U * (uint32_t)8U + i0) * (uint32_t)8U + dotIdx];
      uint32_t i1 = (uint32_t)0U;
      for (; i1 < (uint32_t)8U; i1 += (uint32_t)1U)
        rBrow[i1] = sB[dotIdx * (uint32_t)64U + threadIdx.x % (uint32_t)8U * (uint32_t)8U + i1];
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)8U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
          rchProd[resIdxM * (uint32_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  float_t *t_tile = gC;
  uint32_t resIdxM = (uint32_t)0U;
  while (resIdxM < (uint32_t)8U)
  {
    uint32_t resIdxN = (uint32_t)0U;
    for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
      t_tile[(blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U +
        threadIdx.x / (uint32_t)8U * (uint32_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U +
          threadIdx.x % (uint32_t)8U * (uint32_t)8U + resIdxN]
      =
        beta *
          t_tile[(blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U +
            threadIdx.x / (uint32_t)8U * (uint32_t)8U + resIdxM)
          * cols
          +
            blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U +
              threadIdx.x % (uint32_t)8U * (uint32_t)8U + resIdxN]
        + alpha * rchProd[resIdxM * (uint32_t)8U + resIdxN];
    resIdxM += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rrr_rr(
  float_t alpha,
  float_t beta,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)8U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_2,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)64U,
    (uint32_t)4096U,
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
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)4096U);
  uint32_t num_n_tiles = cols / (uint32_t)128U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (uint32_t)64U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < shared / (uint32_t)8U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    float_t *tileA = gA;
    uint32_t i1 = threadIdx.x;
    for (; i1 < (uint32_t)1024U; i1 += (uint32_t)256U)
      sA[i1] =
        tileA[(mrow * (uint32_t)128U + i1 / (uint32_t)8U) * shared +
          __anf01 * (uint32_t)8U + i1 % (uint32_t)8U];
    float_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)256U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)8U + i / (uint32_t)128U) * cols +
          mcol * (uint32_t)128U + i % (uint32_t)128U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)8U)
    {
      uint32_t i0 = (uint32_t)0U;
      for (; i0 < (uint32_t)8U; i0 += (uint32_t)1U)
        rAcol[i0] = sA[(threadIdx.x / (uint32_t)16U * (uint32_t)8U + i0) * (uint32_t)8U + dotIdx];
      uint32_t i1 = (uint32_t)0U;
      for (; i1 < (uint32_t)8U; i1 += (uint32_t)1U)
        rBrow[i1] = sB[dotIdx * (uint32_t)128U + threadIdx.x % (uint32_t)16U * (uint32_t)8U + i1];
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)8U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
          rchProd[resIdxM * (uint32_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  float_t *t_tile = gC;
  uint32_t resIdxM = (uint32_t)0U;
  while (resIdxM < (uint32_t)8U)
  {
    uint32_t resIdxN = (uint32_t)0U;
    for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
      t_tile[(blockIdx.x / (cols / (uint32_t)128U) * (uint32_t)128U +
        threadIdx.x / (uint32_t)16U * (uint32_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (uint32_t)128U) * (uint32_t)128U +
          threadIdx.x % (uint32_t)16U * (uint32_t)8U + resIdxN]
      =
        beta *
          t_tile[(blockIdx.x / (cols / (uint32_t)128U) * (uint32_t)128U +
            threadIdx.x / (uint32_t)16U * (uint32_t)8U + resIdxM)
          * cols
          +
            blockIdx.x % (cols / (uint32_t)128U) * (uint32_t)128U +
              threadIdx.x % (uint32_t)16U * (uint32_t)8U + resIdxN]
        + alpha * rchProd[resIdxM * (uint32_t)8U + resIdxN];
    resIdxM += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr_rr(
  float_t alpha,
  float_t beta,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)128U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)8U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)128U == (uint32_t)0U);
  KPR_KCALL(__hoisted_3,
    rows / (uint32_t)128U * (cols / (uint32_t)128U),
    (uint32_t)256U,
    (uint32_t)8192U,
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
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)4096U);
  uint32_t num_n_tiles = cols / (uint32_t)128U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  float_t rAcol[8U];
  memset(rAcol, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (uint32_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (uint32_t)64U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < shared / (uint32_t)8U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    float_t *tileA = gA;
    uint32_t i1 = threadIdx.x;
    for (; i1 < (uint32_t)1024U; i1 += (uint32_t)256U)
      sA[i1 % (uint32_t)8U * (uint32_t)128U + i1 / (uint32_t)8U] =
        tileA[(mrow * (uint32_t)128U + i1 / (uint32_t)8U) * shared +
          __anf01 * (uint32_t)8U + i1 % (uint32_t)8U];
    float_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)256U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)8U + i / (uint32_t)128U) * cols +
          mcol * (uint32_t)128U + i % (uint32_t)128U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)8U)
    {
      uint32_t i0 = (uint32_t)0U;
      for (; i0 < (uint32_t)8U; i0 += (uint32_t)1U)
        rAcol[i0] = sA[dotIdx * (uint32_t)128U + threadIdx.x / (uint32_t)16U * (uint32_t)8U + i0];
      uint32_t i1 = (uint32_t)0U;
      for (; i1 < (uint32_t)8U; i1 += (uint32_t)1U)
        rBrow[i1] = sB[dotIdx * (uint32_t)128U + threadIdx.x % (uint32_t)16U * (uint32_t)8U + i1];
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)8U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
          rchProd[resIdxM * (uint32_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  float_t *t_tile = gC;
  uint32_t resIdxM = (uint32_t)0U;
  while (resIdxM < (uint32_t)8U)
  {
    uint32_t resIdxN = (uint32_t)0U;
    for (; resIdxN < (uint32_t)8U; resIdxN += (uint32_t)1U)
      t_tile[(blockIdx.x / (cols / (uint32_t)128U) * (uint32_t)128U +
        threadIdx.x / (uint32_t)16U * (uint32_t)8U + resIdxM)
      * cols
      +
        blockIdx.x % (cols / (uint32_t)128U) * (uint32_t)128U +
          threadIdx.x % (uint32_t)16U * (uint32_t)8U + resIdxN]
      =
        beta *
          t_tile[(blockIdx.x / (cols / (uint32_t)128U) * (uint32_t)128U +
            threadIdx.x / (uint32_t)16U * (uint32_t)8U + resIdxM)
          * cols
          +
            blockIdx.x % (cols / (uint32_t)128U) * (uint32_t)128U +
              threadIdx.x % (uint32_t)16U * (uint32_t)8U + resIdxN]
        + alpha * rchProd[resIdxM * (uint32_t)8U + resIdxN];
    resIdxM += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr_cr(
  float_t alpha,
  float_t beta,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)128U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)8U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)128U == (uint32_t)0U);
  KPR_KCALL(__hoisted_4,
    rows / (uint32_t)128U * (cols / (uint32_t)128U),
    (uint32_t)256U,
    (uint32_t)8192U,
    alpha,
    beta,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

