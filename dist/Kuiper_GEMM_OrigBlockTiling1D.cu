

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
  float_t *sA = (float_t *)KPR_SHMEM_AT(0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT(2048U);
  uint32_t mrow = blockIdx.x / mcols;
  uint32_t mcol = blockIdx.x % mcols;
  float_t cache1d[8U];
  memset(cache1d, 0U, 8U * sizeof (float_t));
  uint32_t bkIdx = 0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    sA[threadIdx.x] =
      gA4[(mrow * 64U + threadIdx.x / 8U) * shared + __anf01 * 8U + threadIdx.x % 8U];
    sB[threadIdx.x] =
      gB4[(__anf01 * 8U + threadIdx.x / 64U) * cols + mcol * 64U + threadIdx.x % 64U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    while (dotIdx < 8U)
    {
      float_t tmpB = sB[dotIdx * 64U + threadIdx.x % 64U];
      uint32_t resIdx = 0U;
      for (; resIdx < 8U; resIdx += 1U)
        cache1d[resIdx] += sA[(threadIdx.x / 64U * 8U + resIdx) * 8U + dotIdx] * tmpB;
      dotIdx += 1U;
    }
    bkIdx += 1U;
  }
  uint32_t resIdx = 0U;
  for (; resIdx < 8U; resIdx += 1U)
    gC4[(mrow * 64U + threadIdx.x / 64U * 8U + resIdx) * cols + mcol * 64U + threadIdx.x % 64U] =
      cache1d[resIdx];
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
  float_t *gA = (float_t *)KPR_GPU_ALLOC(4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC(4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC(4U, rows * cols);
  MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % 64U == 0U);
  KPR_GUARD(shared % 8U == 0U);
  KPR_GUARD(cols % 64U == 0U);
  uint32_t mcols = cols / 64U;
  KPR_KCALL(__hoisted_0,
    rows / 64U * mcols,
    512U,
    4096U,
    shared,
    cols,
    shared / 8U,
    mcols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
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
  float_t *sA = (float_t *)KPR_SHMEM_AT(0U);
  float_t *sB = (float_t *)KPR_SHMEM_AT(2048U);
  uint32_t mrow = blockIdx.x / mcols;
  uint32_t mcol = blockIdx.x % mcols;
  float_t cache1d[8U];
  memset(cache1d, 0U, 8U * sizeof (float_t));
  uint32_t bkIdx = 0U;
  while (bkIdx < mshared)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    sA[threadIdx.x] =
      gA4[(mrow * 64U + threadIdx.x / 8U) * shared + __anf01 * 8U + threadIdx.x % 8U];
    sB[threadIdx.x] =
      gB4[(__anf01 * 8U + threadIdx.x / 64U) * cols + mcol * 64U + threadIdx.x % 64U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    while (dotIdx < 8U)
    {
      float_t tmpB = sB[dotIdx * 64U + threadIdx.x % 64U];
      uint32_t resIdx = 0U;
      for (; resIdx < 8U; resIdx += 1U)
        cache1d[resIdx] += sA[(threadIdx.x / 64U * 8U + resIdx) * 8U + dotIdx] * tmpB;
      dotIdx += 1U;
    }
    bkIdx += 1U;
  }
  uint32_t resIdx = 0U;
  for (; resIdx < 8U; resIdx += 1U)
    gC4[(mrow * 64U + threadIdx.x / 64U * 8U + resIdx) * cols + mcol * 64U + threadIdx.x % 64U] =
      beta *
        gC4[(mrow * 64U + threadIdx.x / 64U * 8U + resIdx) * cols + mcol * 64U + threadIdx.x % 64U]
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
  KPR_GUARD(rows % 64U == 0U);
  KPR_GUARD(shared % 8U == 0U);
  KPR_GUARD(cols % 64U == 0U);
  uint32_t mcols = cols / 64U;
  KPR_KCALL(__hoisted_1,
    rows / 64U * mcols,
    512U,
    4096U,
    alpha,
    beta,
    shared,
    cols,
    shared / 8U,
    mcols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

