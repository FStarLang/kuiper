

#include "Kuiper_GEMM_BlockTiling2D.h"

__global__
/**
  hoisted when extracting matmul_f32_64x64x8_8x8_rrr
*/
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
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    size_t __anf200179 = bkIdx;
    size_t i = threadIdx.x;
    while (i < (size_t)512U)
    {
      sA[i / (size_t)8U * (size_t)8U + i % (size_t)8U] =
        gA4[(mrow * (size_t)64U + i / (size_t)8U) * shared +
          __anf200179 * (size_t)8U + i % (size_t)8U];
      i += (size_t)64U;
    }
    size_t i1 = threadIdx.x;
    while (i1 < (size_t)512U)
    {
      sB[i1 / (size_t)64U * (size_t)64U + i1 % (size_t)64U] =
        gB4[(__anf200179 * (size_t)8U + i1 / (size_t)64U) * cols +
          mcol * (size_t)64U + i1 % (size_t)64U];
      i1 += (size_t)64U;
    }
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)8U)
      {
        rAcol[i0] = sA[(threadIdx.x / (size_t)8U * (size_t)8U + i0) * (size_t)8U + dotIdx];
        i0 += (size_t)1U;
      }
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)8U)
      {
        rBrow[i1] = sB[dotIdx * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + i1];
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)8U)
        {
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t mrow1 = blockIdx.x / mcols;
  size_t mcol1 = blockIdx.x % mcols;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    while (resIdxN < (size_t)8U)
    {
      gC4[(mrow1 * (size_t)64U + threadIdx.x / (size_t)8U * (size_t)8U + resIdxM) * cols +
        mcol1 * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + resIdxN]
      = rchProd[resIdxM * (size_t)8U + resIdxN];
      resIdxN += (size_t)1U;
    }
    resIdxM += (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_64x64x8_8x8_rrr(
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
  size_t mcols = cols / (size_t)64U;
  KPR_KCALL(__hoisted_0,
    rows / (size_t)64U * mcols,
    (size_t)64U,
    (size_t)4096U,
    shared,
    cols,
    shared / (size_t)8U,
    mcols,
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
  hoisted when extracting matmul_f32_32x32x32_32x8_rrr
*/
static void
__hoisted_1(
  size_t shared,
  size_t cols,
  size_t mshared,
  size_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)4096U);
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  float_t rAcol[32U];
  memset(rAcol, 0U, (size_t)32U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[256U];
  memset(rchProd, 0U, (size_t)256U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    size_t __anf200179 = bkIdx;
    size_t i = threadIdx.x;
    while (i < (size_t)1024U)
    {
      sA[i / (size_t)32U * (size_t)32U + i % (size_t)32U] =
        gA4[(mrow * (size_t)32U + i / (size_t)32U) * shared +
          __anf200179 * (size_t)32U + i % (size_t)32U];
      i += (size_t)4U;
    }
    size_t i1 = threadIdx.x;
    while (i1 < (size_t)1024U)
    {
      sB[i1 / (size_t)32U * (size_t)32U + i1 % (size_t)32U] =
        gB4[(__anf200179 * (size_t)32U + i1 / (size_t)32U) * cols +
          mcol * (size_t)32U + i1 % (size_t)32U];
      i1 += (size_t)4U;
    }
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)32U)
    {
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)32U)
      {
        rAcol[i0] = sA[(threadIdx.x / (size_t)4U * (size_t)32U + i0) * (size_t)32U + dotIdx];
        i0 += (size_t)1U;
      }
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)8U)
      {
        rBrow[i1] = sB[dotIdx * (size_t)32U + threadIdx.x % (size_t)4U * (size_t)8U + i1];
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)32U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)8U)
        {
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t mrow1 = blockIdx.x / mcols;
  size_t mcol1 = blockIdx.x % mcols;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)32U)
  {
    size_t resIdxN = (size_t)0U;
    while (resIdxN < (size_t)8U)
    {
      gC4[(mrow1 * (size_t)32U + threadIdx.x / (size_t)4U * (size_t)32U + resIdxM) * cols +
        mcol1 * (size_t)32U + threadIdx.x % (size_t)4U * (size_t)8U + resIdxN]
      = rchProd[resIdxM * (size_t)8U + resIdxN];
      resIdxN += (size_t)1U;
    }
    resIdxM += (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_32x32x32_32x8_rrr(
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
  KPR_KCALL(__hoisted_1,
    rows / (size_t)32U * mcols,
    (size_t)4U,
    (size_t)8192U,
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
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting g_gemm_f32_64x64x8_8x8_rrr
*/
static void
__hoisted_2(
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
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    size_t __anf200179 = bkIdx;
    size_t i = threadIdx.x;
    while (i < (size_t)512U)
    {
      sA[i / (size_t)8U * (size_t)8U + i % (size_t)8U] =
        gA4[(mrow * (size_t)64U + i / (size_t)8U) * shared +
          __anf200179 * (size_t)8U + i % (size_t)8U];
      i += (size_t)64U;
    }
    size_t i1 = threadIdx.x;
    while (i1 < (size_t)512U)
    {
      sB[i1 / (size_t)64U * (size_t)64U + i1 % (size_t)64U] =
        gB4[(__anf200179 * (size_t)8U + i1 / (size_t)64U) * cols +
          mcol * (size_t)64U + i1 % (size_t)64U];
      i1 += (size_t)64U;
    }
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)8U)
      {
        rAcol[i0] = sA[(threadIdx.x / (size_t)8U * (size_t)8U + i0) * (size_t)8U + dotIdx];
        i0 += (size_t)1U;
      }
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)8U)
      {
        rBrow[i1] = sB[dotIdx * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + i1];
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)8U)
        {
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t mrow1 = blockIdx.x / mcols;
  size_t mcol1 = blockIdx.x % mcols;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    while (resIdxN < (size_t)8U)
    {
      size_t vresIdxM = resIdxM;
      size_t vresIdxN = resIdxN;
      gC4[(mrow1 * (size_t)64U + threadIdx.x / (size_t)8U * (size_t)8U + vresIdxM) * cols +
        mcol1 * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + vresIdxN]
      =
        beta *
          gC4[(mrow1 * (size_t)64U + threadIdx.x / (size_t)8U * (size_t)8U + vresIdxM) * cols +
            mcol1 * (size_t)64U + threadIdx.x % (size_t)8U * (size_t)8U + vresIdxN]
        + alpha * rchProd[resIdxM * (size_t)8U + resIdxN];
      resIdxN += (size_t)1U;
    }
    resIdxM += (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rrr(
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
  size_t mcols = cols / (size_t)64U;
  KPR_KCALL(__hoisted_2,
    rows / (size_t)64U * mcols,
    (size_t)64U,
    (size_t)4096U,
    alpha,
    beta,
    shared,
    cols,
    shared / (size_t)8U,
    mcols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f32_128x128x8_8x8_rrr
*/
static void
__hoisted_3(
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
  float_t *sA = (float_t *)KPR_SHMEM_AT((size_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((size_t)4096U);
  size_t mrow = blockIdx.x / mcols;
  size_t mcol = blockIdx.x % mcols;
  float_t rAcol[8U];
  memset(rAcol, 0U, (size_t)8U * sizeof (float_t));
  float_t rBrow[8U];
  memset(rBrow, 0U, (size_t)8U * sizeof (float_t));
  float_t rchProd[64U];
  memset(rchProd, 0U, (size_t)64U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    size_t __anf200179 = bkIdx;
    size_t i = threadIdx.x;
    while (i < (size_t)1024U)
    {
      sA[i / (size_t)8U * (size_t)8U + i % (size_t)8U] =
        gA4[(mrow * (size_t)128U + i / (size_t)8U) * shared +
          __anf200179 * (size_t)8U + i % (size_t)8U];
      i += (size_t)256U;
    }
    size_t i1 = threadIdx.x;
    while (i1 < (size_t)1024U)
    {
      sB[i1 / (size_t)128U * (size_t)128U + i1 % (size_t)128U] =
        gB4[(__anf200179 * (size_t)8U + i1 / (size_t)128U) * cols +
          mcol * (size_t)128U + i1 % (size_t)128U];
      i1 += (size_t)256U;
    }
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)8U)
      {
        rAcol[i0] = sA[(threadIdx.x / (size_t)16U * (size_t)8U + i0) * (size_t)8U + dotIdx];
        i0 += (size_t)1U;
      }
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)8U)
      {
        rBrow[i1] = sB[dotIdx * (size_t)128U + threadIdx.x % (size_t)16U * (size_t)8U + i1];
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)8U)
        {
          rchProd[resIdxM * (size_t)8U + resIdxN] += rAcol[resIdxM] * rBrow[resIdxN];
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t mrow1 = blockIdx.x / mcols;
  size_t mcol1 = blockIdx.x % mcols;
  size_t resIdxM = (size_t)0U;
  while (resIdxM < (size_t)8U)
  {
    size_t resIdxN = (size_t)0U;
    while (resIdxN < (size_t)8U)
    {
      size_t vresIdxM = resIdxM;
      size_t vresIdxN = resIdxN;
      gC4[(mrow1 * (size_t)128U + threadIdx.x / (size_t)16U * (size_t)8U + vresIdxM) * cols +
        mcol1 * (size_t)128U + threadIdx.x % (size_t)16U * (size_t)8U + vresIdxN]
      =
        beta *
          gC4[(mrow1 * (size_t)128U + threadIdx.x / (size_t)16U * (size_t)8U + vresIdxM) * cols +
            mcol1 * (size_t)128U + threadIdx.x % (size_t)16U * (size_t)8U + vresIdxN]
        + alpha * rchProd[resIdxM * (size_t)8U + resIdxN];
      resIdxN += (size_t)1U;
    }
    resIdxM += (size_t)1U;
  }
}

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr(
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
  size_t mcols = cols / (size_t)128U;
  KPR_KCALL(__hoisted_3,
    rows / (size_t)128U * mcols,
    (size_t)256U,
    (size_t)8192U,
    alpha,
    beta,
    shared,
    cols,
    shared / (size_t)8U,
    mcols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

