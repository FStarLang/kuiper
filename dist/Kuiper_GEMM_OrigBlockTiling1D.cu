

#include "Kuiper_GEMM_OrigBlockTiling1D.h"

__global__
/**
  hoisted when extracting matmul_f32_tiles64x8_8x64_rc8_rrr
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
  float_t cache1d[8U];
  memset(cache1d, 0U, (size_t)8U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    size_t __anf184786 = bkIdx;
    ((float_t *)sA)[threadIdx.x / (size_t)8U * (size_t)8U + threadIdx.x % (size_t)8U] =
      ((float_t *)gA4)[(mrow * (size_t)64U + threadIdx.x / (size_t)8U) * shared +
        __anf184786 * (size_t)8U + threadIdx.x % (size_t)8U];
    ((float_t *)sB)[threadIdx.x / (size_t)64U * (size_t)64U + threadIdx.x % (size_t)64U] =
      ((float_t *)gB4)[(__anf184786 * (size_t)8U + threadIdx.x / (size_t)64U) * cols +
        mcol * (size_t)64U + threadIdx.x % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      float_t tmpB = ((float_t *)sB)[dotIdx * (size_t)64U + threadIdx.x % (size_t)64U];
      size_t resIdx = (size_t)0U;
      while (resIdx < (size_t)8U)
      {
        cache1d[resIdx] +=
          ((float_t *)sA)[(threadIdx.x / (size_t)64U * (size_t)8U + resIdx) * (size_t)8U + dotIdx] *
            tmpB;
        resIdx += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t resIdx = (size_t)0U;
  while (resIdx < (size_t)8U)
  {
    size_t vresIdx = resIdx;
    KRML_HOST_IGNORE(((float_t *)gC4)[(mrow * (size_t)64U +
        threadIdx.x / (size_t)64U * (size_t)8U + vresIdx)
      * cols
      + mcol * (size_t)64U + threadIdx.x % (size_t)64U]);
    ((float_t *)gC4)[(mrow * (size_t)64U + threadIdx.x / (size_t)64U * (size_t)8U + vresIdx) * cols
    + mcol * (size_t)64U + threadIdx.x % (size_t)64U]
    = cache1d[resIdx];
    resIdx += (size_t)1U;
  }
}

float_t
*Kuiper_GEMM_OrigBlockTiling1D_matmul_f32_tiles64x8_8x64_rc8_rrr(
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
    (size_t)512U,
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
  hoisted when extracting g_gemm_f32_tiles64x8_8x64_rc8_rrr
*/
static void
__hoisted_1(
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
  float_t cache1d[8U];
  memset(cache1d, 0U, (size_t)8U * sizeof (float_t));
  size_t bkIdx = (size_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    size_t __anf184786 = bkIdx;
    ((float_t *)sA)[threadIdx.x / (size_t)8U * (size_t)8U + threadIdx.x % (size_t)8U] =
      ((float_t *)gA4)[(mrow * (size_t)64U + threadIdx.x / (size_t)8U) * shared +
        __anf184786 * (size_t)8U + threadIdx.x % (size_t)8U];
    ((float_t *)sB)[threadIdx.x / (size_t)64U * (size_t)64U + threadIdx.x % (size_t)64U] =
      ((float_t *)gB4)[(__anf184786 * (size_t)8U + threadIdx.x / (size_t)64U) * cols +
        mcol * (size_t)64U + threadIdx.x % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)8U)
    {
      float_t tmpB = ((float_t *)sB)[dotIdx * (size_t)64U + threadIdx.x % (size_t)64U];
      size_t resIdx = (size_t)0U;
      while (resIdx < (size_t)8U)
      {
        cache1d[resIdx] +=
          ((float_t *)sA)[(threadIdx.x / (size_t)64U * (size_t)8U + resIdx) * (size_t)8U + dotIdx] *
            tmpB;
        resIdx += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t resIdx = (size_t)0U;
  while (resIdx < (size_t)8U)
  {
    size_t vresIdx = resIdx;
    ((float_t *)gC4)[(mrow * (size_t)64U + threadIdx.x / (size_t)64U * (size_t)8U + vresIdx) * cols
    + mcol * (size_t)64U + threadIdx.x % (size_t)64U]
    =
      beta *
        ((float_t *)gC4)[(mrow * (size_t)64U + threadIdx.x / (size_t)64U * (size_t)8U + vresIdx) *
          cols
        + mcol * (size_t)64U + threadIdx.x % (size_t)64U]
      + alpha * cache1d[resIdx];
    resIdx += (size_t)1U;
  }
}

void
Kuiper_GEMM_OrigBlockTiling1D_g_gemm_f32_tiles64x8_8x64_rc8_rrr(
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
  KPR_KCALL(__hoisted_1,
    rows / (size_t)64U * mcols,
    (size_t)512U,
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

