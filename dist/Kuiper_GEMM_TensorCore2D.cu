

#include "Kuiper_GEMM_TensorCore2D.h"

typedef uint32_t has_vec_cpy;

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_4x4
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)16U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)4U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)4U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (uint32_t)16U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)16U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)16U;
      uint32_t col = i2 % (uint32_t)16U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)64U) + __anf01 * (uint32_t)16U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)16U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)64U;
      uint32_t col = i % (uint32_t)64U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)16U) + mcol * (uint32_t)64U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)64U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)1U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)4U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)16U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)16U * (i0 * (uint32_t)16U)),
          (uint32_t)16U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)4U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)64U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
            + i1 * (uint32_t)16U),
          (uint32_t)64U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)4U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)4U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)4U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)4U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)4U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)4U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
            blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
          + cols * (i * (uint32_t)16U)
          + j * (uint32_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x4(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_0,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)32U,
    (uint32_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_32x8x16_1x2
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          32,
          8,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          32,
          8,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
        (uint32_t)2U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)2U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)32U;
      uint32_t col = i2 % (uint32_t)32U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)32U) + __anf01 * (uint32_t)32U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)32U + col + k] = local[k];
      i2 += (uint32_t)64U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)32U;
      uint32_t col = i % (uint32_t)32U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)32U) + mcol * (uint32_t)32U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)32U + col + k] = local[k];
      i += (uint32_t)64U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)1U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)32U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)32U * (i0 * (uint32_t)32U)),
          (uint32_t)32U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)2U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)32U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U
            + i1 * (uint32_t)8U),
          (uint32_t)32U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)1U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)2U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)1U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)2U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
            blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)32U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U
          + cols * (i * (uint32_t)32U)
          + j * (uint32_t)8U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_32x8x16_1x2(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  KPR_KCALL(__hoisted_1,
    rows / (uint32_t)32U * (cols / (uint32_t)32U),
    (uint32_t)64U,
    (uint32_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_8x32x16_2x1
*/
static void __hoisted_2(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          8,
          32,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          8,
          32,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
        (uint32_t)2U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)2U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)32U;
      uint32_t col = i2 % (uint32_t)32U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)32U) + __anf01 * (uint32_t)32U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)32U + col + k] = local[k];
      i2 += (uint32_t)64U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)32U;
      uint32_t col = i % (uint32_t)32U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)32U) + mcol * (uint32_t)32U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)32U + col + k] = local[k];
      i += (uint32_t)64U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)2U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)32U * (i0 * (uint32_t)8U)),
          (uint32_t)32U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)1U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)32U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
            + i1 * (uint32_t)32U),
          (uint32_t)32U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)2U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)1U)
        {
          auto& acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)2U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)1U)
    {
      auto& __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
            blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
          + cols * (i * (uint32_t)8U)
          + j * (uint32_t)32U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_8x32x16_2x1(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  KPR_KCALL(__hoisted_2,
    rows / (uint32_t)32U * (cols / (uint32_t)32U),
    (uint32_t)64U,
    (uint32_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x8x16_32x8x16
*/
static void __hoisted_3(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)1024U);
  uint32_t num_k_tiles = shared / (uint32_t)16U;
  uint32_t num_n_tiles = cols / (uint32_t)8U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          32,
          8,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          32,
          8,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
        (uint32_t)1U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)1U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)512U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)16U;
      uint32_t col = i2 % (uint32_t)16U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)32U) + __anf01 * (uint32_t)16U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)16U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)128U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)8U;
      uint32_t col = i % (uint32_t)8U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)16U) + mcol * (uint32_t)8U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)8U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)1U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)1U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)16U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)16U * (i0 * (uint32_t)32U)),
          (uint32_t)16U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)1U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)8U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)8U
            + i1 * (uint32_t)8U),
          (uint32_t)8U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)1U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)1U)
        {
          auto& acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)1U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)1U)
    {
      auto& __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)8U) * (uint32_t)32U) +
            blockIdx.x % (cols / (uint32_t)8U) * (uint32_t)8U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)8U
          + cols * (i * (uint32_t)32U)
          + j * (uint32_t)8U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x8x16_32x8x16(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)8U == (uint32_t)0U);
  KPR_KCALL(__hoisted_3,
    rows / (uint32_t)32U * (cols / (uint32_t)8U),
    (uint32_t)32U,
    (uint32_t)1280U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_8x32x16_8x32x16
*/
static void __hoisted_4(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          8,
          32,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          8,
          32,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
        (uint32_t)1U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)1U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)32U;
      uint32_t col = i2 % (uint32_t)32U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)32U) + __anf01 * (uint32_t)32U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)32U + col + k] = local[k];
      i2 += (uint32_t)128U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)32U;
      uint32_t col = i % (uint32_t)32U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)32U) + mcol * (uint32_t)32U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)32U + col + k] = local[k];
      i += (uint32_t)128U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)1U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)32U * (i0 * (uint32_t)8U)),
          (uint32_t)32U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)1U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)32U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
            + i1 * (uint32_t)32U),
          (uint32_t)32U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)1U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)1U)
        {
          auto& acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)1U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)1U)
    {
      auto& __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
            blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
          + cols * (i * (uint32_t)8U)
          + j * (uint32_t)32U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_8x32x16_8x32x16(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  KPR_KCALL(__hoisted_4,
    rows / (uint32_t)32U * (cols / (uint32_t)32U),
    (uint32_t)128U,
    (uint32_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_16x16x16
*/
static void __hoisted_5(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)512U);
  uint32_t num_k_tiles = shared / (uint32_t)16U;
  uint32_t num_n_tiles = cols / (uint32_t)16U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)1U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (uint32_t)1U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)1U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)256U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)16U;
      uint32_t col = i2 % (uint32_t)16U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)16U) + __anf01 * (uint32_t)16U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)16U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)256U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)16U;
      uint32_t col = i % (uint32_t)16U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)16U) + mcol * (uint32_t)16U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)16U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)1U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)1U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)16U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)16U * (i0 * (uint32_t)16U)),
          (uint32_t)16U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)1U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)16U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)16U
            + i1 * (uint32_t)16U),
          (uint32_t)16U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)1U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)1U)
        {
          auto& acc_frag = accFrags[resIdxM + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)1U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)1U)
    {
      auto& __anf2 = accFrags[i + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)16U) * (uint32_t)16U) +
            blockIdx.x % (cols / (uint32_t)16U) * (uint32_t)16U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)16U
          + cols * (i * (uint32_t)16U)
          + j * (uint32_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_16x16x16(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  KPR_KCALL(__hoisted_5,
    rows / (uint32_t)16U * (cols / (uint32_t)16U),
    (uint32_t)32U,
    (uint32_t)1024U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_4x4
*/
static void __hoisted_6(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)4U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)4U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (uint32_t)16U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)16U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)64U;
      uint32_t col = i2 % (uint32_t)64U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)64U) + __anf01 * (uint32_t)64U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)64U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)64U;
      uint32_t col = i % (uint32_t)64U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)64U) + mcol * (uint32_t)64U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)64U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)4U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)64U * (i0 * (uint32_t)16U)),
          (uint32_t)64U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)4U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)64U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
            + i1 * (uint32_t)16U),
          (uint32_t)64U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)4U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)4U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)4U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)4U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)4U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)4U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
            blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
          + cols * (i * (uint32_t)16U)
          + j * (uint32_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x4(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_6,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)32U,
    (uint32_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_32x8x16_2x8
*/
static void __hoisted_7(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          32,
          8,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          32,
          8,
          16,
          wmma::row_major),
        (uint32_t)8U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16),
        (uint32_t)16U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)16U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)64U;
      uint32_t col = i2 % (uint32_t)64U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)64U) + __anf01 * (uint32_t)64U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)64U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)64U;
      uint32_t col = i % (uint32_t)64U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)64U) + mcol * (uint32_t)64U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)64U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)2U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)64U * (i0 * (uint32_t)32U)),
          (uint32_t)64U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)8U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)64U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
            + i1 * (uint32_t)8U),
          (uint32_t)64U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)2U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)8U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)8U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)2U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)8U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)8U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
            blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
          + cols * (i * (uint32_t)32U)
          + j * (uint32_t)8U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_32x8x16_2x8(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_7,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)32U,
    (uint32_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_8x32x16_8x2
*/
static void __hoisted_8(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          8,
          32,
          16,
          wmma::row_major),
        (uint32_t)8U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          8,
          32,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16),
        (uint32_t)16U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)16U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)64U;
      uint32_t col = i2 % (uint32_t)64U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)64U) + __anf01 * (uint32_t)64U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)64U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)64U;
      uint32_t col = i % (uint32_t)64U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)64U) + mcol * (uint32_t)64U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)64U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)8U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)64U * (i0 * (uint32_t)8U)),
          (uint32_t)64U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)2U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)64U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
            + i1 * (uint32_t)32U),
          (uint32_t)64U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)8U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)2U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)8U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)2U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
            blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)64U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)64U
          + cols * (i * (uint32_t)8U)
          + j * (uint32_t)32U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_8x32x16_8x2(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_8,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)32U,
    (uint32_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_16x16x16_2x2
*/
static void __hoisted_9(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (uint32_t)4U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)4U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)32U;
      uint32_t col = i2 % (uint32_t)32U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)32U) + __anf01 * (uint32_t)32U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)32U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)32U;
      uint32_t col = i % (uint32_t)32U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)32U) + mcol * (uint32_t)32U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)32U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)2U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)32U * (i0 * (uint32_t)16U)),
          (uint32_t)32U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)2U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)32U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
            + i1 * (uint32_t)16U),
          (uint32_t)32U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)2U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)2U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)2U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)2U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
            blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
          + cols * (i * (uint32_t)16U)
          + j * (uint32_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_16x16x16_2x2(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  KPR_KCALL(__hoisted_9,
    rows / (uint32_t)32U * (cols / (uint32_t)32U),
    (uint32_t)32U,
    (uint32_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_2x2
*/
static void __hoisted_10(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16),
        (uint32_t)4U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)4U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (half_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)64U;
      uint32_t col = i2 % (uint32_t)64U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)64U) + __anf01 * (uint32_t)64U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)64U + col + k] = local[k];
      i2 += (uint32_t)128U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)4096U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)64U;
      uint32_t col = i % (uint32_t)64U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)64U) + mcol * (uint32_t)64U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)64U + col + k] = local[k];
      i += (uint32_t)128U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)2U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)32U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)64U * (i0 * (uint32_t)16U)),
          (uint32_t)64U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)2U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)64U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)32U
            + i1 * (uint32_t)16U),
          (uint32_t)64U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)2U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)2U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)2U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)2U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
            blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)32U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)32U
          + cols * (i * (uint32_t)16U)
          + j * (uint32_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x2(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)64U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)64U == (uint32_t)0U);
  KPR_KCALL(__hoisted_10,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)128U,
    (uint32_t)16384U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f32_32x32x32_16x16x16_2x2
*/
static void __hoisted_11(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, float_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_a,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  bFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE(half,
          wmma::matrix_b,
          16,
          16,
          16,
          wmma::row_major),
        (uint32_t)2U));
  auto&
  accFrags =
    KPR_INIT(KPR_ARRAY_FRAGMENT_TYPE(KPR_FRAGMENT_TYPE_C(float, wmma::accumulator, 16, 16, 16),
        (uint32_t)4U));
  uint32_t fi = (uint32_t)0U;
  for (; fi < (uint32_t)4U; fi += (uint32_t)1U)
    wmma::fill_fragment(accFrags[fi], (float_t)0.0f);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i2 = threadIdx.x * (uint32_t)8U;
    while (i2 < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i2 / (uint32_t)32U;
      uint32_t col = i2 % (uint32_t)32U;
      vec_memcpy(local,
        tileA + shared * (mrow * (uint32_t)32U) + __anf01 * (uint32_t)32U + shared * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sA[row * (uint32_t)32U + col + k] = local[k];
      i2 += (uint32_t)32U * (uint32_t)8U;
    }
    half_t *tileB = gB;
    uint32_t i = threadIdx.x * (uint32_t)8U;
    while (i < (uint32_t)1024U)
    {
      half_t local[8U];
      memset(local, 0U, (uint32_t)8U * sizeof (half_t));
      uint32_t row = i / (uint32_t)32U;
      uint32_t col = i % (uint32_t)32U;
      vec_memcpy(local,
        tileB + cols * (__anf01 * (uint32_t)32U) + mcol * (uint32_t)32U + cols * row + col);
      uint32_t k = (uint32_t)0U;
      for (; k < (uint32_t)8U; k += (uint32_t)1U)
        sB[row * (uint32_t)32U + col + k] = local[k];
      i += (uint32_t)32U * (uint32_t)8U;
    }
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf04 = dotIdx;
      half_t *tile_for_tc_a_tiles = sA;
      uint32_t i0 = (uint32_t)0U;
      while (i0 < (uint32_t)2U)
      {
        auto& __anf1 = aFrags[i0];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_a_tiles,
            (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U) +
              __anf04 * (uint32_t)16U
            + (uint32_t)32U * (i0 * (uint32_t)16U)),
          (uint32_t)32U);
        i0 += (uint32_t)1U;
      }
      uint32_t __anf05 = dotIdx;
      half_t *tile_for_tc_b_tiles = sB;
      uint32_t i1 = (uint32_t)0U;
      while (i1 < (uint32_t)2U)
      {
        auto& __anf1 = bFrags[i1];
        wmma::load_matrix_sync(__anf1,
          kpr_offset(tile_for_tc_b_tiles,
            (uint32_t)32U * (__anf05 * (uint32_t)16U) +
              threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
            + i1 * (uint32_t)16U),
          (uint32_t)32U);
        i1 += (uint32_t)1U;
      }
      uint32_t resIdxM = (uint32_t)0U;
      while (resIdxM < (uint32_t)2U)
      {
        uint32_t resIdxN = (uint32_t)0U;
        while (resIdxN < (uint32_t)2U)
        {
          auto& acc_frag = accFrags[resIdxM * (uint32_t)2U + resIdxN];
          wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN], acc_frag);
          resIdxN += (uint32_t)1U;
        }
        resIdxM += (uint32_t)1U;
      }
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  uint32_t i = (uint32_t)0U;
  while (i < (uint32_t)2U)
  {
    uint32_t j = (uint32_t)0U;
    while (j < (uint32_t)2U)
    {
      auto& __anf2 = accFrags[i * (uint32_t)2U + j];
      wmma::store_matrix_sync(kpr_offset(gC,
          cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
            blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
          + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U)
          + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U
          + cols * (i * (uint32_t)16U)
          + j * (uint32_t)16U),
        __anf2,
        cols,
        wmma::mem_row_major);
      j += (uint32_t)1U;
    }
    i += (uint32_t)1U;
  }
}

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f32_32x32x32_16x16x16_2x2(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  half_t *gA,
  half_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  KPR_KCALL(__hoisted_11,
    rows / (uint32_t)32U * (cols / (uint32_t)32U),
    (uint32_t)32U,
    (uint32_t)4096U,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

