

#include "Kuiper_GEMM_OrigBlockTiling1D.h"

__global__
/**
  hoisted when extracting matmul_f32_tiles64x8_8x64_rc8_rrr
*/
static void
__hoisted_0(
  uint32_t shared,
  uint32_t cols,
  uint32_t mshared,
  uint32_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t mrow = blockIdx.x / mcols;
  uint32_t mcol = blockIdx.x % mcols;
  float_t cache1d[8U];
  memset(cache1d, 0U, (uint32_t)8U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    sA[threadIdx.x] =
      gA4[(mrow * (uint32_t)64U + threadIdx.x / (uint32_t)8U) * shared +
        __anf01 * (uint32_t)8U + threadIdx.x % (uint32_t)8U];
    sB[threadIdx.x] =
      gB4[(__anf01 * (uint32_t)8U + threadIdx.x / (uint32_t)64U) * cols +
        mcol * (uint32_t)64U + threadIdx.x % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)8U)
    {
      float_t tmpB = sB[dotIdx * (uint32_t)64U + threadIdx.x % (uint32_t)64U];
      uint32_t resIdx = (uint32_t)0U;
      for (; resIdx < (uint32_t)8U; resIdx += (uint32_t)1U)
        cache1d[resIdx] +=
          sA[(threadIdx.x / (uint32_t)64U * (uint32_t)8U + resIdx) * (uint32_t)8U + dotIdx] * tmpB;
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t resIdx = (uint32_t)0U;
  for (; resIdx < (uint32_t)8U; resIdx += (uint32_t)1U)
    gC4[(mrow * (uint32_t)64U + threadIdx.x / (uint32_t)64U * (uint32_t)8U + resIdx) * cols +
      mcol * (uint32_t)64U + threadIdx.x % (uint32_t)64U]
    = cache1d[resIdx];
}

float_t
*Kuiper_GEMM_OrigBlockTiling1D_matmul_f32_tiles64x8_8x64_rc8_rrr(
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
  uint32_t mcols = cols / (uint32_t)64U;
  KPR_KCALL(__hoisted_0,
    rows / (uint32_t)64U * mcols,
    (uint32_t)512U,
    (uint32_t)4096U,
    shared,
    cols,
    shared / (uint32_t)8U,
    mcols,
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
  hoisted when extracting g_gemm_f32_tiles64x8_8x64_rc8_rrr
*/
static void
__hoisted_1(
  float_t alpha,
  float_t beta,
  uint32_t shared,
  uint32_t cols,
  uint32_t mshared,
  uint32_t mcols,
  float_t *gA4,
  float_t *gB4,
  float_t *gC4
)
{
  float_t *sA = (float_t *)KPR_SHMEM_AT((uint32_t)0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t mrow = blockIdx.x / mcols;
  uint32_t mcol = blockIdx.x % mcols;
  float_t cache1d[8U];
  memset(cache1d, 0U, (uint32_t)8U * sizeof (float_t));
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    sA[threadIdx.x] =
      gA4[(mrow * (uint32_t)64U + threadIdx.x / (uint32_t)8U) * shared +
        __anf01 * (uint32_t)8U + threadIdx.x % (uint32_t)8U];
    sB[threadIdx.x] =
      gB4[(__anf01 * (uint32_t)8U + threadIdx.x / (uint32_t)64U) * cols +
        mcol * (uint32_t)64U + threadIdx.x % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)8U)
    {
      float_t tmpB = sB[dotIdx * (uint32_t)64U + threadIdx.x % (uint32_t)64U];
      uint32_t resIdx = (uint32_t)0U;
      for (; resIdx < (uint32_t)8U; resIdx += (uint32_t)1U)
        cache1d[resIdx] +=
          sA[(threadIdx.x / (uint32_t)64U * (uint32_t)8U + resIdx) * (uint32_t)8U + dotIdx] * tmpB;
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t resIdx = (uint32_t)0U;
  for (; resIdx < (uint32_t)8U; resIdx += (uint32_t)1U)
    gC4[(mrow * (uint32_t)64U + threadIdx.x / (uint32_t)64U * (uint32_t)8U + resIdx) * cols +
      mcol * (uint32_t)64U + threadIdx.x % (uint32_t)64U]
    =
      beta *
        gC4[(mrow * (uint32_t)64U + threadIdx.x / (uint32_t)64U * (uint32_t)8U + resIdx) * cols +
          mcol * (uint32_t)64U + threadIdx.x % (uint32_t)64U]
      + alpha * cache1d[resIdx];
}

void
Kuiper_GEMM_OrigBlockTiling1D_g_gemm_f32_tiles64x8_8x64_rc8_rrr(
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
  uint32_t mcols = cols / (uint32_t)64U;
  KPR_KCALL(__hoisted_1,
    rows / (uint32_t)64U * mcols,
    (uint32_t)512U,
    (uint32_t)4096U,
    alpha,
    beta,
    shared,
    cols,
    shared / (uint32_t)8U,
    mcols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

