

#include "Kuiper_GEMM_TensorCore.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16_rrr
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(2048U);
  uint32_t num_k_tiles = shared / 16U;
  uint32_t num_n_tiles = cols / 64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto&
  bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 4U * 16U)
      + threadIdx.x / 32U % 4U * 16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 1024U; i0 += 512U)
      sA[i0] = tileA[(mrow * 64U + i0 / 16U) * shared + __anf01 * 16U + i0 % 16U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 1024U; i += 512U)
      sB[i] = tileB[(__anf01 * 16U + i / 64U) * cols + mcol * 64U + i % 64U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 1U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 16U * (threadIdx.x / 32U / 4U * 16U) + dotIdx * 16U),
        16U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 64U * (__anf05 * 16U) + threadIdx.x / 32U % 4U * 16U),
        64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 4U * 16U)
      + threadIdx.x / 32U % 4U * 16U),
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
  KPR_GUARD(rows % 64U == 0U);
  KPR_GUARD(shared % 16U == 0U);
  KPR_GUARD(cols % 64U == 0U);
  KPR_KCALL(__hoisted_0, rows / 64U * (cols / 64U), 512U, 4096U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_32x8x16_rrr
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(2048U);
  uint32_t num_k_tiles = shared / 32U;
  uint32_t num_n_tiles = cols / 32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto& aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 32, 8, 16, wmma::row_major));
  auto& bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 32, 8, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 4U * 32U)
      + threadIdx.x / 32U % 4U * 8U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 1024U; i0 += 128U)
      sA[i0] = tileA[(mrow * 32U + i0 / 32U) * shared + __anf01 * 32U + i0 % 32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 1024U; i += 128U)
      sB[i] = tileB[(__anf01 * 32U + i / 32U) * cols + mcol * 32U + i % 32U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 2U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 32U * (threadIdx.x / 32U / 4U * 32U) + dotIdx * 16U),
        32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 32U * (__anf05 * 16U) + threadIdx.x / 32U % 4U * 8U),
        32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 4U * 32U)
      + threadIdx.x / 32U % 4U * 8U),
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
  KPR_GUARD(rows % 32U == 0U);
  KPR_GUARD(shared % 32U == 0U);
  KPR_GUARD(cols % 32U == 0U);
  KPR_KCALL(__hoisted_1, rows / 32U * (cols / 32U), 128U, 4096U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_8x32x16_rrr
*/
static void __hoisted_2(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(2048U);
  uint32_t num_k_tiles = shared / 32U;
  uint32_t num_n_tiles = cols / 32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto& aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 8, 32, 16, wmma::row_major));
  auto& bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 8, 32, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 1U * 8U)),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 1024U; i0 += 128U)
      sA[i0] = tileA[(mrow * 32U + i0 / 32U) * shared + __anf01 * 32U + i0 % 32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 1024U; i += 128U)
      sB[i] = tileB[(__anf01 * 32U + i / 32U) * cols + mcol * 32U + i % 32U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 2U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 32U * (threadIdx.x / 32U / 1U * 8U) + dotIdx * 16U),
        32U);
      wmma::load_matrix_sync(bFrag, kpr_offset(b_tile, 32U * (__anf05 * 16U) + 0U * 32U), 32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 1U * 8U)),
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
  KPR_GUARD(rows % 32U == 0U);
  KPR_GUARD(shared % 32U == 0U);
  KPR_GUARD(cols % 32U == 0U);
  KPR_KCALL(__hoisted_2, rows / 32U * (cols / 32U), 128U, 4096U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x8x16_32x8x16_rrr
*/
static void __hoisted_3(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(1024U);
  uint32_t num_k_tiles = shared / 16U;
  uint32_t num_n_tiles = cols / 8U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto& aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 32, 8, 16, wmma::row_major));
  auto& bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 32, 8, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 8U) * 32U) + blockIdx.x % (cols / 8U) * 8U +
        cols * (threadIdx.x / 32U / 1U * 32U)),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 512U; i0 += 32U)
      sA[i0] = tileA[(mrow * 32U + i0 / 16U) * shared + __anf01 * 16U + i0 % 16U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 128U; i += 32U)
      sB[i] = tileB[(__anf01 * 16U + i / 8U) * cols + mcol * 8U + i % 8U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 1U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 16U * (threadIdx.x / 32U / 1U * 32U) + dotIdx * 16U),
        16U);
      wmma::load_matrix_sync(bFrag, kpr_offset(b_tile, 8U * (__anf05 * 16U) + 0U * 8U), 8U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 8U) * 32U) + blockIdx.x % (cols / 8U) * 8U +
        cols * (threadIdx.x / 32U / 1U * 32U)),
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
  KPR_GUARD(rows % 32U == 0U);
  KPR_GUARD(shared % 16U == 0U);
  KPR_GUARD(cols % 8U == 0U);
  KPR_KCALL(__hoisted_3, rows / 32U * (cols / 8U), 32U, 1280U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_8x32x16_8x32x16_rrr
*/
static void __hoisted_4(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(2048U);
  uint32_t num_k_tiles = shared / 32U;
  uint32_t num_n_tiles = cols / 32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto& aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 8, 32, 16, wmma::row_major));
  auto& bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 8, 32, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 1U * 8U)),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 1024U; i0 += 128U)
      sA[i0] = tileA[(mrow * 32U + i0 / 32U) * shared + __anf01 * 32U + i0 % 32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 1024U; i += 128U)
      sB[i] = tileB[(__anf01 * 32U + i / 32U) * cols + mcol * 32U + i % 32U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 2U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 32U * (threadIdx.x / 32U / 1U * 8U) + dotIdx * 16U),
        32U);
      wmma::load_matrix_sync(bFrag, kpr_offset(b_tile, 32U * (__anf05 * 16U) + 0U * 32U), 32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 1U * 8U)),
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
  KPR_GUARD(rows % 32U == 0U);
  KPR_GUARD(shared % 32U == 0U);
  KPR_GUARD(cols % 32U == 0U);
  KPR_KCALL(__hoisted_4, rows / 32U * (cols / 32U), 128U, 4096U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16_rrr
*/
static void __hoisted_5(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(8192U);
  uint32_t num_k_tiles = shared / 64U;
  uint32_t num_n_tiles = cols / 64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto&
  bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 4U * 16U)
      + threadIdx.x / 32U % 4U * 16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 4096U; i0 += 512U)
      sA[i0] = tileA[(mrow * 64U + i0 / 64U) * shared + __anf01 * 64U + i0 % 64U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 4096U; i += 512U)
      sB[i] = tileB[(__anf01 * 64U + i / 64U) * cols + mcol * 64U + i % 64U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 4U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 64U * (threadIdx.x / 32U / 4U * 16U) + dotIdx * 16U),
        64U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 64U * (__anf05 * 16U) + threadIdx.x / 32U % 4U * 16U),
        64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 4U * 16U)
      + threadIdx.x / 32U % 4U * 16U),
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
  KPR_GUARD(rows % 64U == 0U);
  KPR_GUARD(shared % 64U == 0U);
  KPR_GUARD(cols % 64U == 0U);
  KPR_KCALL(__hoisted_5, rows / 64U * (cols / 64U), 512U, 16384U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_32x8x16_rrr
*/
static void __hoisted_6(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(8192U);
  uint32_t num_k_tiles = shared / 64U;
  uint32_t num_n_tiles = cols / 64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto& aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 32, 8, 16, wmma::row_major));
  auto& bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 32, 8, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 32, 8, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 8U * 32U)
      + threadIdx.x / 32U % 8U * 8U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 4096U; i0 += 512U)
      sA[i0] = tileA[(mrow * 64U + i0 / 64U) * shared + __anf01 * 64U + i0 % 64U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 4096U; i += 512U)
      sB[i] = tileB[(__anf01 * 64U + i / 64U) * cols + mcol * 64U + i % 64U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 4U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 64U * (threadIdx.x / 32U / 8U * 32U) + dotIdx * 16U),
        64U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 64U * (__anf05 * 16U) + threadIdx.x / 32U % 8U * 8U),
        64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 8U * 32U)
      + threadIdx.x / 32U % 8U * 8U),
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
  KPR_GUARD(rows % 64U == 0U);
  KPR_GUARD(shared % 64U == 0U);
  KPR_GUARD(cols % 64U == 0U);
  KPR_KCALL(__hoisted_6, rows / 64U * (cols / 64U), 512U, 16384U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_8x32x16_rrr
*/
static void __hoisted_7(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(8192U);
  uint32_t num_k_tiles = shared / 64U;
  uint32_t num_n_tiles = cols / 64U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto& aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 8, 32, 16, wmma::row_major));
  auto& bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 8, 32, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 8, 32, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 2U * 8U)
      + threadIdx.x / 32U % 2U * 32U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 4096U; i0 += 512U)
      sA[i0] = tileA[(mrow * 64U + i0 / 64U) * shared + __anf01 * 64U + i0 % 64U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 4096U; i += 512U)
      sB[i] = tileB[(__anf01 * 64U + i / 64U) * cols + mcol * 64U + i % 64U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 4U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 64U * (threadIdx.x / 32U / 2U * 8U) + dotIdx * 16U),
        64U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 64U * (__anf05 * 16U) + threadIdx.x / 32U % 2U * 32U),
        64U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 64U) * 64U) + blockIdx.x % (cols / 64U) * 64U +
        cols * (threadIdx.x / 32U / 2U * 8U)
      + threadIdx.x / 32U % 2U * 32U),
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
  KPR_GUARD(rows % 64U == 0U);
  KPR_GUARD(shared % 64U == 0U);
  KPR_GUARD(cols % 64U == 0U);
  KPR_KCALL(__hoisted_7, rows / 64U * (cols / 64U), 512U, 16384U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_16x16x16_rrr
*/
static void __hoisted_8(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(2048U);
  uint32_t num_k_tiles = shared / 32U;
  uint32_t num_n_tiles = cols / 32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto&
  bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 2U * 16U)
      + threadIdx.x / 32U % 2U * 16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 1024U; i0 += 128U)
      sA[i0] = tileA[(mrow * 32U + i0 / 32U) * shared + __anf01 * 32U + i0 % 32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 1024U; i += 128U)
      sB[i] = tileB[(__anf01 * 32U + i / 32U) * cols + mcol * 32U + i % 32U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 2U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 32U * (threadIdx.x / 32U / 2U * 16U) + dotIdx * 16U),
        32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 32U * (__anf05 * 16U) + threadIdx.x / 32U % 2U * 16U),
        32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 2U * 16U)
      + threadIdx.x / 32U % 2U * 16U),
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
  KPR_GUARD(rows % 32U == 0U);
  KPR_GUARD(shared % 32U == 0U);
  KPR_GUARD(cols % 32U == 0U);
  KPR_KCALL(__hoisted_8, rows / 32U * (cols / 32U), 128U, 4096U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_16x16x16_rrr
*/
static void __hoisted_9(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, half_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(512U);
  uint32_t num_k_tiles = shared / 16U;
  uint32_t num_n_tiles = cols / 16U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto&
  bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 16U) * 16U) + blockIdx.x % (cols / 16U) * 16U +
        cols * (threadIdx.x / 32U / 1U * 16U)),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 256U; i0 += 32U)
      sA[i0] = tileA[(mrow * 16U + i0 / 16U) * shared + __anf01 * 16U + i0 % 16U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 256U; i += 32U)
      sB[i] = tileB[(__anf01 * 16U + i / 16U) * cols + mcol * 16U + i % 16U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 1U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 16U * (threadIdx.x / 32U / 1U * 16U) + dotIdx * 16U),
        16U);
      wmma::load_matrix_sync(bFrag, kpr_offset(b_tile, 16U * (__anf05 * 16U) + 0U * 16U), 16U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 16U) * 16U) + blockIdx.x % (cols / 16U) * 16U +
        cols * (threadIdx.x / 32U / 1U * 16U)),
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
  KPR_GUARD(rows % 16U == 0U);
  KPR_GUARD(shared % 16U == 0U);
  KPR_GUARD(cols % 16U == 0U);
  KPR_KCALL(__hoisted_9, rows / 16U * (cols / 16U), 32U, 1024U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f32_32x32x32_16x16x16_rrr
*/
static void __hoisted_10(uint32_t shared, uint32_t cols, half_t *gA, half_t *gB, float_t *gC)
{
  half_t *sA = (half_t *)KPR_SHMEM_AT(0U);
  half_t *sB = (half_t *)KPR_SHMEM_AT(2048U);
  uint32_t num_k_tiles = shared / 32U;
  uint32_t num_n_tiles = cols / 32U;
  uint32_t mrow = blockIdx.x / num_n_tiles;
  uint32_t mcol = blockIdx.x % num_n_tiles;
  auto&
  aFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto&
  bFrag = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& accumFrag = KPR_INIT(KPR_FRAGMENT_TYPE_C(float, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(accumFrag,
    kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 2U * 16U)
      + threadIdx.x / 32U % 2U * 16U),
    cols,
    wmma::mem_row_major);
  uint32_t bkIdx = 0U;
  for (; bkIdx < num_k_tiles; bkIdx += 1U)
  {
    __syncthreads();
    uint32_t __anf01 = bkIdx;
    half_t *tileA = gA;
    uint32_t i0 = threadIdx.x;
    for (; i0 < 1024U; i0 += 128U)
      sA[i0] = tileA[(mrow * 32U + i0 / 32U) * shared + __anf01 * 32U + i0 % 32U];
    half_t *tileB = gB;
    uint32_t i = threadIdx.x;
    for (; i < 1024U; i += 128U)
      sB[i] = tileB[(__anf01 * 32U + i / 32U) * cols + mcol * 32U + i % 32U];
    __syncthreads();
    uint32_t dotIdx = 0U;
    for (; dotIdx < 2U; dotIdx += 1U)
    {
      uint32_t __anf05 = dotIdx;
      half_t *b_tile = sB;
      wmma::load_matrix_sync(aFrag,
        kpr_offset(sA, 32U * (threadIdx.x / 32U / 2U * 16U) + dotIdx * 16U),
        32U);
      wmma::load_matrix_sync(bFrag,
        kpr_offset(b_tile, 32U * (__anf05 * 16U) + threadIdx.x / 32U % 2U * 16U),
        32U);
      wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
    }
  }
  wmma::store_matrix_sync(kpr_offset(gC,
      cols * (blockIdx.x / (cols / 32U) * 32U) + blockIdx.x % (cols / 32U) * 32U +
        cols * (threadIdx.x / 32U / 2U * 16U)
      + threadIdx.x / 32U % 2U * 16U),
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
  KPR_GUARD(rows % 32U == 0U);
  KPR_GUARD(shared % 32U == 0U);
  KPR_GUARD(cols % 32U == 0U);
  KPR_KCALL(__hoisted_10, rows / 32U * (cols / 32U), 128U, 4096U, shared, cols, gA, gB, gC);
  cudaDeviceSynchronize();
}

