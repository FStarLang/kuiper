

#include "Kuiper_GEMM_TensorCore.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_rrr
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)16U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)1024U; i0 += (uint32_t)512U)
      sA[i0] =
        tileA[(mrow * (uint32_t)64U + i0 / (uint32_t)16U) * shared +
          __anf01 * (uint32_t)16U + i0 % (uint32_t)16U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)512U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)16U + i / (uint32_t)64U) * cols +
          mcol * (uint32_t)64U + i % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)16U * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)16U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)16U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)64U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)16U),
        (uint32_t)64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)16U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x16_16x16x16_rrr(
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
    (uint32_t)512U,
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
  hoisted when extracting g_gemm_f16_f16_32x32x32_32x8x16_rrr
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 32, 8, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 32, 8, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)32U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)8U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)1024U; i0 += (uint32_t)128U)
      sA[i0] =
        tileA[(mrow * (uint32_t)32U + i0 / (uint32_t)32U) * shared +
          __anf01 * (uint32_t)32U + i0 % (uint32_t)32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)128U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)32U + i / (uint32_t)32U) * cols +
          mcol * (uint32_t)32U + i % (uint32_t)32U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)32U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)32U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)8U),
        (uint32_t)32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)32U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)8U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_32x8x16_rrr(
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
  hoisted when extracting g_gemm_f16_f16_32x32x32_8x32x16_rrr
*/
static void __hoisted_2(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 8, 32, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 8, 32, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)1024U; i0 += (uint32_t)128U)
      sA[i0] =
        tileA[(mrow * (uint32_t)32U + i0 / (uint32_t)32U) * shared +
          __anf01 * (uint32_t)32U + i0 % (uint32_t)32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)128U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)32U + i / (uint32_t)32U) * cols +
          mcol * (uint32_t)32U + i % (uint32_t)32U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)32U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U),
        (uint32_t)32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_8x32x16_rrr(
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
  hoisted when extracting g_gemm_f16_f16_32x8x16_32x8x16_rrr
*/
static void __hoisted_3(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)1024U);
  uint32_t num_k_tiles = shared / (uint32_t)16U;
  uint32_t num_n_tiles = cols / (uint32_t)8U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 32, 8, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 32, 8, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)8U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)8U) * (uint32_t)8U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)8U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)512U; i0 += (uint32_t)32U)
      sA[i0] =
        tileA[(mrow * (uint32_t)32U + i0 / (uint32_t)16U) * shared +
          __anf01 * (uint32_t)16U + i0 % (uint32_t)16U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)128U; i += (uint32_t)32U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)16U + i / (uint32_t)8U) * cols +
          mcol * (uint32_t)8U + i % (uint32_t)8U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)16U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)16U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)8U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)8U),
        (uint32_t)8U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)8U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)8U) * (uint32_t)8U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)32U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)8U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x8x16_32x8x16_rrr(
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
  hoisted when extracting g_gemm_f16_f16_8x32x16_8x32x16_rrr
*/
static void __hoisted_4(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 8, 32, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 8, 32, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)1024U; i0 += (uint32_t)128U)
      sA[i0] =
        tileA[(mrow * (uint32_t)32U + i0 / (uint32_t)32U) * shared +
          __anf01 * (uint32_t)32U + i0 % (uint32_t)32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)128U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)32U + i / (uint32_t)32U) * cols +
          mcol * (uint32_t)32U + i % (uint32_t)32U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)32U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U),
        (uint32_t)32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)8U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)32U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_8x32x16_8x32x16_rrr(
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
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_rrr
*/
static void __hoisted_5(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)4096U; i0 += (uint32_t)512U)
      sA[i0] =
        tileA[(mrow * (uint32_t)64U + i0 / (uint32_t)64U) * shared +
          __anf01 * (uint32_t)64U + i0 % (uint32_t)64U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)4096U; i += (uint32_t)512U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)64U + i / (uint32_t)64U) * cols +
          mcol * (uint32_t)64U + i % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)16U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)64U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)64U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)16U),
        (uint32_t)64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)4U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)4U * (uint32_t)16U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16_rrr(
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
  KPR_KCALL(__hoisted_5,
    rows / (uint32_t)64U * (cols / (uint32_t)64U),
    (uint32_t)512U,
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
  hoisted when extracting g_gemm_f16_f16_64x64x64_32x8x16_rrr
*/
static void __hoisted_6(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 32, 8, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 32, 8, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)8U * (uint32_t)32U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)8U * (uint32_t)8U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)4096U; i0 += (uint32_t)512U)
      sA[i0] =
        tileA[(mrow * (uint32_t)64U + i0 / (uint32_t)64U) * shared +
          __anf01 * (uint32_t)64U + i0 % (uint32_t)64U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)4096U; i += (uint32_t)512U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)64U + i / (uint32_t)64U) * cols +
          mcol * (uint32_t)64U + i % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)8U * (uint32_t)32U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)64U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)64U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)8U * (uint32_t)8U),
        (uint32_t)64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)8U * (uint32_t)32U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)8U * (uint32_t)8U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_32x8x16_rrr(
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
    (uint32_t)512U,
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
  hoisted when extracting g_gemm_f16_f16_64x64x64_8x32x16_rrr
*/
static void __hoisted_7(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)8192U);
  uint32_t num_k_tiles = shared / (uint32_t)64U;
  uint32_t num_n_tiles = cols / (uint32_t)64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 8, 32, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 8, 32, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)8U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)32U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)4096U; i0 += (uint32_t)512U)
      sA[i0] =
        tileA[(mrow * (uint32_t)64U + i0 / (uint32_t)64U) * shared +
          __anf01 * (uint32_t)64U + i0 % (uint32_t)64U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)4096U; i += (uint32_t)512U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)64U + i / (uint32_t)64U) * cols +
          mcol * (uint32_t)64U + i % (uint32_t)64U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)4U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)64U * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)8U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)64U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)64U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)32U),
        (uint32_t)64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)64U) * (uint32_t)64U) +
        blockIdx.x % (cols / (uint32_t)64U) * (uint32_t)64U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)8U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)32U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_8x32x16_rrr(
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
    (uint32_t)512U,
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
  hoisted when extracting g_gemm_f16_f16_32x32x32_16x16x16_rrr
*/
static void __hoisted_8(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)1024U; i0 += (uint32_t)128U)
      sA[i0] =
        tileA[(mrow * (uint32_t)32U + i0 / (uint32_t)32U) * shared +
          __anf01 * (uint32_t)32U + i0 % (uint32_t)32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)128U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)32U + i / (uint32_t)32U) * cols +
          mcol * (uint32_t)32U + i % (uint32_t)32U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)16U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)32U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U),
        (uint32_t)32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_16x16x16_rrr(
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
  KPR_KCALL(__hoisted_8,
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
  hoisted when extracting g_gemm_f16_f16_16x16x16_16x16x16_rrr
*/
static void __hoisted_9(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)512U);
  uint32_t num_k_tiles = shared / (uint32_t)16U;
  uint32_t num_n_tiles = cols / (uint32_t)16U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)16U) * (uint32_t)16U) +
        blockIdx.x % (cols / (uint32_t)16U) * (uint32_t)16U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)256U; i0 += (uint32_t)32U)
      sA[i0] =
        tileA[(mrow * (uint32_t)16U + i0 / (uint32_t)16U) * shared +
          __anf01 * (uint32_t)16U + i0 % (uint32_t)16U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)256U; i += (uint32_t)32U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)16U + i / (uint32_t)16U) * cols +
          mcol * (uint32_t)16U + i % (uint32_t)16U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)16U * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)16U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)16U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)16U),
        (uint32_t)16U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)16U) * (uint32_t)16U) +
        blockIdx.x % (cols / (uint32_t)16U) * (uint32_t)16U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)1U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)1U * (uint32_t)16U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_16x16x16_16x16x16_rrr(
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
  KPR_KCALL(__hoisted_9,
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
  hoisted when extracting g_gemm_f16_f32_32x32x32_16x16x16_rrr
*/
static void __hoisted_10(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, float_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT((uint32_t)0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT((uint32_t)2048U);
  uint32_t num_k_tiles = shared / (uint32_t)32U;
  uint32_t num_n_tiles = cols / (uint32_t)32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(float, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = (uint32_t)0U;
  while (bkIdx < num_k_tiles)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < (uint32_t)1024U; i0 += (uint32_t)128U)
      sA[i0] =
        tileA[(mrow * (uint32_t)32U + i0 / (uint32_t)32U) * shared +
          __anf01 * (uint32_t)32U + i0 % (uint32_t)32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < (uint32_t)1024U; i += (uint32_t)128U)
      sB[i] =
        tileB[(__anf01 * (uint32_t)32U + i / (uint32_t)32U) * cols +
          mcol * (uint32_t)32U + i % (uint32_t)32U];
    __syncthreads();
    uint32_t dotIdx = (uint32_t)0U;
    while (dotIdx < (uint32_t)2U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA,
          (uint32_t)32U * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)16U) +
            dotIdx * (uint32_t)16U),
        (uint32_t)32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile,
          (uint32_t)32U * (__anf05 * (uint32_t)16U) +
            threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U),
        (uint32_t)32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
      dotIdx += (uint32_t)1U;
    }
    bkIdx += (uint32_t)1U;
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / (uint32_t)32U) * (uint32_t)32U) +
        blockIdx.x % (cols / (uint32_t)32U) * (uint32_t)32U
      + cols * (threadIdx.x / (uint32_t)32U / (uint32_t)2U * (uint32_t)16U)
      + threadIdx.x / (uint32_t)32U % (uint32_t)2U * (uint32_t)16U),
    accumFrag,
    cols,
    wmma::mem_row_major);
}

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f32_32x32x32_16x16x16_rrr(
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
  KPR_KCALL(__hoisted_10,
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

