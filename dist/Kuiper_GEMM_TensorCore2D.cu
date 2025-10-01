

#include "Kuiper_GEMM_TensorCore2D.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x4_rrr
*/
static void __hoisted_0(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_k_tiles = shared / (size_t)16U;
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)4U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)4U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (size_t)16U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)16U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)1024U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)64U + i2 / (size_t)16U) * shared +
          __anf01 * (size_t)16U + i2 % (size_t)16U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)16U + i / (size_t)64U) * cols +
          mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)1U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)4U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)16U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U) +
              __anf04 * (size_t)16U
            + (size_t)16U * (i0 * (size_t)16U)),
          (size_t)16U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)4U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)64U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
            + i1 * (size_t)16U),
          (size_t)64U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)4U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)4U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)4U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)4U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)4U)
    {
      auto __anf2 = accFrags[i * (size_t)4U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)64U) * (size_t)64U) +
            blockIdx.x % (cols / (size_t)64U) * (size_t)64U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
          + cols * (i * (size_t)16U)
          + j * (size_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x4_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_0,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)32U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_32x8x16_1x2_rrr
*/
static void __hoisted_1(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_k_tiles = shared / (size_t)32U;
  size_t num_n_tiles = cols / (size_t)32U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          32,
          8,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          32,
          8,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
        (size_t)2U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)2U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)1024U; i2 += (size_t)64U)
      sA[i2] =
        tileA[(mrow * (size_t)32U + i2 / (size_t)32U) * shared +
          __anf01 * (size_t)32U + i2 % (size_t)32U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)64U)
      sB[i] =
        tileB[(__anf01 * (size_t)32U + i / (size_t)32U) * cols +
          mcol * (size_t)32U + i % (size_t)32U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)2U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)1U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)32U * (threadIdx.x / (size_t)32U / (size_t)2U * (size_t)32U) +
              __anf04 * (size_t)16U
            + (size_t)32U * (i0 * (size_t)32U)),
          (size_t)32U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)2U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)32U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)2U * (size_t)16U
            + i1 * (size_t)8U),
          (size_t)32U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)1U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)2U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)1U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)2U)
    {
      auto __anf2 = accFrags[i * (size_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)32U) * (size_t)32U) +
            blockIdx.x % (cols / (size_t)32U) * (size_t)32U
          + cols * (threadIdx.x / (size_t)32U / (size_t)2U * (size_t)32U)
          + threadIdx.x / (size_t)32U % (size_t)2U * (size_t)16U
          + cols * (i * (size_t)32U)
          + j * (size_t)8U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_32x8x16_1x2_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  KPR_KCALL(__hoisted_1,
    rows / (size_t)32U * (cols / (size_t)32U),
    (size_t)64U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_8x32x16_2x1_rrr
*/
static void __hoisted_2(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_k_tiles = shared / (size_t)32U;
  size_t num_n_tiles = cols / (size_t)32U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          8,
          32,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          8,
          32,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
        (size_t)2U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)2U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)1024U; i2 += (size_t)64U)
      sA[i2] =
        tileA[(mrow * (size_t)32U + i2 / (size_t)32U) * shared +
          __anf01 * (size_t)32U + i2 % (size_t)32U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)64U)
      sB[i] =
        tileB[(__anf01 * (size_t)32U + i / (size_t)32U) * cols +
          mcol * (size_t)32U + i % (size_t)32U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)2U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)2U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)32U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)16U) +
              __anf04 * (size_t)16U
            + (size_t)32U * (i0 * (size_t)8U)),
          (size_t)32U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)1U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)32U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
            + i1 * (size_t)32U),
          (size_t)32U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)2U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)1U)
        {
          auto acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)2U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)1U)
    {
      auto __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)32U) * (size_t)32U) +
            blockIdx.x % (cols / (size_t)32U) * (size_t)32U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)16U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
          + cols * (i * (size_t)8U)
          + j * (size_t)32U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_8x32x16_2x1_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  KPR_KCALL(__hoisted_2,
    rows / (size_t)32U * (cols / (size_t)32U),
    (size_t)64U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x8x16_32x8x16_rrr
*/
static void __hoisted_3(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)1024U);
  size_t num_k_tiles = shared / (size_t)16U;
  size_t num_n_tiles = cols / (size_t)8U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          32,
          8,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          32,
          8,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
        (size_t)1U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)1U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)512U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)32U + i2 / (size_t)16U) * shared +
          __anf01 * (size_t)16U + i2 % (size_t)16U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)128U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)16U + i / (size_t)8U) * cols + mcol * (size_t)8U + i % (size_t)8U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)1U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)1U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)16U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)32U) +
              __anf04 * (size_t)16U
            + (size_t)16U * (i0 * (size_t)32U)),
          (size_t)16U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)1U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)8U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)8U
            + i1 * (size_t)8U),
          (size_t)8U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)1U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)1U)
        {
          auto acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)1U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)1U)
    {
      auto __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)8U) * (size_t)32U) +
            blockIdx.x % (cols / (size_t)8U) * (size_t)8U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)32U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)8U
          + cols * (i * (size_t)32U)
          + j * (size_t)8U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x8x16_32x8x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)8U == (size_t)0U);
  KPR_KCALL(__hoisted_3,
    rows / (size_t)32U * (cols / (size_t)8U),
    (size_t)32U,
    (size_t)1280U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_8x32x16_8x32x16_rrr
*/
static void __hoisted_4(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_k_tiles = shared / (size_t)32U;
  size_t num_n_tiles = cols / (size_t)32U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          8,
          32,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          8,
          32,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
        (size_t)1U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)1U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)1024U; i2 += (size_t)128U)
      sA[i2] =
        tileA[(mrow * (size_t)32U + i2 / (size_t)32U) * shared +
          __anf01 * (size_t)32U + i2 % (size_t)32U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)128U)
      sB[i] =
        tileB[(__anf01 * (size_t)32U + i / (size_t)32U) * cols +
          mcol * (size_t)32U + i % (size_t)32U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)2U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)1U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)32U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)8U) +
              __anf04 * (size_t)16U
            + (size_t)32U * (i0 * (size_t)8U)),
          (size_t)32U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)1U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)32U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
            + i1 * (size_t)32U),
          (size_t)32U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)1U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)1U)
        {
          auto acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)1U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)1U)
    {
      auto __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)32U) * (size_t)32U) +
            blockIdx.x % (cols / (size_t)32U) * (size_t)32U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)8U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
          + cols * (i * (size_t)8U)
          + j * (size_t)32U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_8x32x16_8x32x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  KPR_KCALL(__hoisted_4,
    rows / (size_t)32U * (cols / (size_t)32U),
    (size_t)128U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_16x16x16_rrr
*/
static void __hoisted_5(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)512U);
  size_t num_k_tiles = shared / (size_t)16U;
  size_t num_n_tiles = cols / (size_t)16U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)1U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (size_t)1U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)1U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)256U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)16U + i2 / (size_t)16U) * shared +
          __anf01 * (size_t)16U + i2 % (size_t)16U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)256U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)16U + i / (size_t)16U) * cols +
          mcol * (size_t)16U + i % (size_t)16U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)1U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)1U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)16U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)16U) +
              __anf04 * (size_t)16U
            + (size_t)16U * (i0 * (size_t)16U)),
          (size_t)16U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)1U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)16U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)16U
            + i1 * (size_t)16U),
          (size_t)16U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)1U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)1U)
        {
          auto acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)1U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)1U)
    {
      auto __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)16U) * (size_t)16U) +
            blockIdx.x % (cols / (size_t)16U) * (size_t)16U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)16U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)16U
          + cols * (i * (size_t)16U)
          + j * (size_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_16x16x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)16U == (size_t)0U);
  KPR_GUARD(shared % (size_t)16U == (size_t)0U);
  KPR_GUARD(cols % (size_t)16U == (size_t)0U);
  KPR_KCALL(__hoisted_5,
    rows / (size_t)16U * (cols / (size_t)16U),
    (size_t)32U,
    (size_t)1024U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x4_rrr
*/
static void __hoisted_6(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)8192U);
  size_t num_k_tiles = shared / (size_t)64U;
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)4U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)4U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (size_t)16U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)16U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)4096U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)64U + i2 / (size_t)64U) * shared +
          __anf01 * (size_t)64U + i2 % (size_t)64U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)4096U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)64U + i / (size_t)64U) * cols +
          mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)4U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)4U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)64U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U) +
              __anf04 * (size_t)16U
            + (size_t)64U * (i0 * (size_t)16U)),
          (size_t)64U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)4U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)64U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
            + i1 * (size_t)16U),
          (size_t)64U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)4U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)4U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)4U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)4U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)4U)
    {
      auto __anf2 = accFrags[i * (size_t)4U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)64U) * (size_t)64U) +
            blockIdx.x % (cols / (size_t)64U) * (size_t)64U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
          + cols * (i * (size_t)16U)
          + j * (size_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x4_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)64U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_6,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)32U,
    (size_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_32x8x16_2x8_rrr
*/
static void __hoisted_7(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)8192U);
  size_t num_k_tiles = shared / (size_t)64U;
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          32,
          8,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          32,
          8,
          16,
          wmma::row_major),
        (size_t)8U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
        (size_t)16U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)16U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)4096U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)64U + i2 / (size_t)64U) * shared +
          __anf01 * (size_t)64U + i2 % (size_t)64U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)4096U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)64U + i / (size_t)64U) * cols +
          mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)4U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)2U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)64U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U) +
              __anf04 * (size_t)16U
            + (size_t)64U * (i0 * (size_t)32U)),
          (size_t)64U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)8U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)64U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
            + i1 * (size_t)8U),
          (size_t)64U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)2U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)8U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)8U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)2U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)8U)
    {
      auto __anf2 = accFrags[i * (size_t)8U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)64U) * (size_t)64U) +
            blockIdx.x % (cols / (size_t)64U) * (size_t)64U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
          + cols * (i * (size_t)32U)
          + j * (size_t)8U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_32x8x16_2x8_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)64U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_7,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)32U,
    (size_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_8x32x16_8x2_rrr
*/
static void __hoisted_8(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)8192U);
  size_t num_k_tiles = shared / (size_t)64U;
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          8,
          32,
          16,
          wmma::row_major),
        (size_t)8U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          8,
          32,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
        (size_t)16U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)16U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)4096U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)64U + i2 / (size_t)64U) * shared +
          __anf01 * (size_t)64U + i2 % (size_t)64U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)4096U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)64U + i / (size_t)64U) * cols +
          mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)4U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)8U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)64U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U) +
              __anf04 * (size_t)16U
            + (size_t)64U * (i0 * (size_t)8U)),
          (size_t)64U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)2U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)64U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
            + i1 * (size_t)32U),
          (size_t)64U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)8U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)2U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)8U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)2U)
    {
      auto __anf2 = accFrags[i * (size_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)64U) * (size_t)64U) +
            blockIdx.x % (cols / (size_t)64U) * (size_t)64U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)64U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)64U
          + cols * (i * (size_t)8U)
          + j * (size_t)32U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_8x32x16_8x2_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)64U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_8,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)32U,
    (size_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_16x16x16_2x2_rrr
*/
static void __hoisted_9(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_k_tiles = shared / (size_t)32U;
  size_t num_n_tiles = cols / (size_t)32U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (size_t)4U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)4U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)1024U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)32U + i2 / (size_t)32U) * shared +
          __anf01 * (size_t)32U + i2 % (size_t)32U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)32U + i / (size_t)32U) * cols +
          mcol * (size_t)32U + i % (size_t)32U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)2U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)2U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)32U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)32U) +
              __anf04 * (size_t)16U
            + (size_t)32U * (i0 * (size_t)16U)),
          (size_t)32U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)2U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)32U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
            + i1 * (size_t)16U),
          (size_t)32U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)2U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)2U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)2U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)2U)
    {
      auto __anf2 = accFrags[i * (size_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)32U) * (size_t)32U) +
            blockIdx.x % (cols / (size_t)32U) * (size_t)32U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)32U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
          + cols * (i * (size_t)16U)
          + j * (size_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_16x16x16_2x2_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  KPR_KCALL(__hoisted_9,
    rows / (size_t)32U * (cols / (size_t)32U),
    (size_t)32U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x2_rrr
*/
static void __hoisted_10(size_t shared, size_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)8192U);
  size_t num_k_tiles = shared / (size_t)64U;
  size_t num_n_tiles = cols / (size_t)64U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (size_t)4U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)4U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)4096U; i2 += (size_t)128U)
      sA[i2] =
        tileA[(mrow * (size_t)64U + i2 / (size_t)64U) * shared +
          __anf01 * (size_t)64U + i2 % (size_t)64U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)4096U; i += (size_t)128U)
      sB[i] =
        tileB[(__anf01 * (size_t)64U + i / (size_t)64U) * cols +
          mcol * (size_t)64U + i % (size_t)64U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)4U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)2U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)64U * (threadIdx.x / (size_t)32U / (size_t)2U * (size_t)32U) +
              __anf04 * (size_t)16U
            + (size_t)64U * (i0 * (size_t)16U)),
          (size_t)64U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)2U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)64U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)2U * (size_t)32U
            + i1 * (size_t)16U),
          (size_t)64U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)2U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)2U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)2U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)2U)
    {
      auto __anf2 = accFrags[i * (size_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)64U) * (size_t)64U) +
            blockIdx.x % (cols / (size_t)64U) * (size_t)64U
          + cols * (threadIdx.x / (size_t)32U / (size_t)2U * (size_t)32U)
          + threadIdx.x / (size_t)32U % (size_t)2U * (size_t)32U
          + cols * (i * (size_t)16U)
          + j * (size_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x2_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (size_t)64U == (size_t)0U);
  KPR_GUARD(shared % (size_t)64U == (size_t)0U);
  KPR_GUARD(cols % (size_t)64U == (size_t)0U);
  KPR_KCALL(__hoisted_10,
    rows / (size_t)64U * (cols / (size_t)64U),
    (size_t)128U,
    (size_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f32_32x32x32_16x16x16_2x2_rrr
*/
static void __hoisted_11(size_t shared, size_t cols, half_t *gA, half_t *gB, float_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((size_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((size_t)2048U);
  size_t num_k_tiles = shared / (size_t)32U;
  size_t num_n_tiles = cols / (size_t)32U;
  size_t mrow = blockIdx.x / num_n_tiles;
  size_t mcol = blockIdx.x % num_n_tiles;
  auto
  *aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (size_t)2U));
  auto
  *accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(float, wmma::accumulator, 16, 16, 16),
        (size_t)4U));
  size_t fi = (size_t)0U;
  for (; fi < (size_t)4U; fi += (size_t)1U)
    wmma::fill_fragment(accFrags[fi], (float_t)0.0f);
  size_t bkIdx = (size_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    size_t __anf01 = bkIdx;
    half_t *tileA = gA;
    size_t i2 = threadIdx.x;
    for (; i2 < (size_t)1024U; i2 += (size_t)32U)
      sA[i2] =
        tileA[(mrow * (size_t)32U + i2 / (size_t)32U) * shared +
          __anf01 * (size_t)32U + i2 % (size_t)32U];
    half_t *tileB = gB;
    size_t i = threadIdx.x;
    for (; i < (size_t)1024U; i += (size_t)32U)
      sB[i] =
        tileB[(__anf01 * (size_t)32U + i / (size_t)32U) * cols +
          mcol * (size_t)32U + i % (size_t)32U];
    __syncthreads();
    size_t dotIdx = (size_t)0U;
    while (dotIdx < (size_t)2U)
    {
      size_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      size_t i0 = (size_t)0U;
      while (i0 < (size_t)2U)
      {
        auto __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (size_t)32U * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)32U) +
              __anf04 * (size_t)16U
            + (size_t)32U * (i0 * (size_t)16U)),
          (size_t)32U);
        i0 += (size_t)1U;
      }
      size_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      size_t i1 = (size_t)0U;
      while (i1 < (size_t)2U)
      {
        auto __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (size_t)32U * (__anf05 * (size_t)16U) +
              threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
            + i1 * (size_t)16U),
          (size_t)32U);
        i1 += (size_t)1U;
      }
      size_t resIdxM = (size_t)0U;
      while (resIdxM < (size_t)2U)
      {
        size_t resIdxN = (size_t)0U;
        while (resIdxN < (size_t)2U)
        {
          auto acc_frag = accFrags[resIdxM * (size_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (size_t)1U;
        }
        resIdxM += (size_t)1U;
      }
      dotIdx += (size_t)1U;
    }
    bkIdx += (size_t)1U;
  }
  size_t i = (size_t)0U;
  while (i < (size_t)2U)
  {
    size_t j = (size_t)0U;
    while (j < (size_t)2U)
    {
      auto __anf2 = accFrags[i * (size_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (size_t)32U) * (size_t)32U) +
            blockIdx.x % (cols / (size_t)32U) * (size_t)32U
          + cols * (threadIdx.x / (size_t)32U / (size_t)1U * (size_t)32U)
          + threadIdx.x / (size_t)32U % (size_t)1U * (size_t)32U
          + cols * (i * (size_t)16U)
          + j * (size_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (size_t)1U;
    }
    i += (size_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f32_32x32x32_16x16x16_2x2_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (size_t)32U == (size_t)0U);
  KPR_GUARD(shared % (size_t)32U == (size_t)0U);
  KPR_GUARD(cols % (size_t)32U == (size_t)0U);
  KPR_KCALL(__hoisted_11,
    rows / (size_t)32U * (cols / (size_t)32U),
    (size_t)32U,
    (size_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

