

#include "Kuiper_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x8_8x8_rr
*/
static void
__hoisted_0(
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
    uint32_t i1 = threadIdx.x * (uint32_t)4U;
    while (i1 < (uint32_t)512U)
    {
      float_t local[4U];
      memset(local, 0U, (uint32_t)4U * sizeof (float_t));
      uint32_t row = i1 / (uint32_t)8U;
      uint32_t col = i1 % (uint32_t)8U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)64U) + __anf01 * (uint32_t)8U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)4U; k += (uint32_t)1U)
        sA[row * (uint32_t)8U + col + k] = local[k];
      i1 += (uint32_t)256U;
    }
    float_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)4U;
    while (i < (uint32_t)512U)
    {
      float_t local[4U];
      memset(local, 0U, (uint32_t)4U * sizeof (float_t));
      uint32_t row = i / (uint32_t)64U;
      uint32_t col = i % (uint32_t)64U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)8U) + mcol * (uint32_t)64U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)4U; k += (uint32_t)1U)
        sB[row * (uint32_t)64U + col + k] = local[k];
      i += (uint32_t)256U;
    }
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rr(
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
  KPR_KCALL(__hoisted_0,
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
  hoisted when extracting g_gemm_f32_128x128x8_8x8_rr
*/
static void
__hoisted_1(
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
    uint32_t i1 = threadIdx.x * (uint32_t)4U;
    while (i1 < (uint32_t)1024U)
    {
      float_t local[4U];
      memset(local, 0U, (uint32_t)4U * sizeof (float_t));
      uint32_t row = i1 / (uint32_t)8U;
      uint32_t col = i1 % (uint32_t)8U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)128U) + __anf01 * (uint32_t)8U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)4U; k += (uint32_t)1U)
        sA[row * (uint32_t)8U + col + k] = local[k];
      i1 += (uint32_t)1024U;
    }
    float_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)4U;
    while (i < (uint32_t)1024U)
    {
      float_t local[4U];
      memset(local, 0U, (uint32_t)4U * sizeof (float_t));
      uint32_t row = i / (uint32_t)128U;
      uint32_t col = i % (uint32_t)128U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)8U) + mcol * (uint32_t)128U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)4U; k += (uint32_t)1U)
        sB[row * (uint32_t)128U + col + k] = local[k];
      i += (uint32_t)1024U;
    }
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rr(
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
  KPR_KCALL(__hoisted_1,
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
  hoisted when extracting g_gemm_f32_128x128x8_8x8_cr
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
    uint32_t i1 = threadIdx.x * (uint32_t)4U;
    while (i1 < (uint32_t)1024U)
    {
      float_t local[4U];
      memset(local, 0U, (uint32_t)4U * sizeof (float_t));
      uint32_t row = i1 / (uint32_t)8U;
      uint32_t col = i1 % (uint32_t)8U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)128U) + __anf01 * (uint32_t)8U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)4U; k += (uint32_t)1U)
        sA[(col + k) * (uint32_t)128U + row] = local[k];
      i1 += (uint32_t)1024U;
    }
    float_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)4U;
    while (i < (uint32_t)1024U)
    {
      float_t local[4U];
      memset(local, 0U, (uint32_t)4U * sizeof (float_t));
      uint32_t row = i / (uint32_t)128U;
      uint32_t col = i % (uint32_t)128U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)8U) + mcol * (uint32_t)128U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)4U; k += (uint32_t)1U)
        sB[row * (uint32_t)128U + col + k] = local[k];
      i += (uint32_t)1024U;
    }
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
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_cr(
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
  KPR_KCALL(__hoisted_2,
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

