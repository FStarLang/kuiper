
#include "Klas_GEMM_TensorCore.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x16_16x16x16
*/
static void
__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                 blockIdx.x % (cols / 64U) * 64U +
                 cols * (threadIdx.x / 32U / 4U) * 16U +
                 threadIdx.x / 32U % 4U * 16U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (16U * (threadIdx.x / 32U / 4U) * 16U + dotIdx * 16U),
                16U);
            wmma::load_matrix_sync(bFrag,
                sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 4U * 16U),
                64U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                                     blockIdx.x % (cols / 64U) * 64U +
                                     cols * (threadIdx.x / 32U / 4U) * 16U +
                                     threadIdx.x / 32U % 4U * 16U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_32x8x16
*/
static void
__hoisted_g_gemm_f16_f16_32x32x32_32x8x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 32U, 8U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 32U, 8U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 32U, 8U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                 blockIdx.x % (cols / 32U) * 32U +
                 cols * (threadIdx.x / 32U / 4U) * 32U +
                 threadIdx.x / 32U % 4U * 8U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 32U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 32U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (32U * (threadIdx.x / 32U / 4U) * 32U + dotIdx * 16U),
                32U);
            wmma::load_matrix_sync(bFrag,
                sarB + (32U * __anf02 * 16U + threadIdx.x / 32U % 4U * 8U),
                32U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                                     blockIdx.x % (cols / 32U) * 32U +
                                     cols * (threadIdx.x / 32U / 4U) * 32U +
                                     threadIdx.x / 32U % 4U * 8U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_8x32x16
*/
static void
__hoisted_g_gemm_f16_f16_32x32x32_8x32x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 8U, 32U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 8U, 32U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 8U, 32U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                 blockIdx.x % (cols / 32U) * 32U +
                 cols * (threadIdx.x / 32U) * 8U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 32U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 32U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (32U * (threadIdx.x / 32U) * 8U + dotIdx * 16U), 32U);
            wmma::load_matrix_sync(bFrag, sarB + 32U * __anf02 * 16U, 32U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                                     blockIdx.x % (cols / 32U) * 32U +
                                     cols * (threadIdx.x / 32U) * 8U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x8x16_32x8x16
*/
static void
__hoisted_g_gemm_f16_f16_32x8x16_32x8x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(1024U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 8U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 32U, 8U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 32U, 8U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 32U, 8U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 8U)) * 32U +
                 blockIdx.x % (cols / 8U) * 8U +
                 cols * (threadIdx.x / 32U) * 32U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 512U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 32U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 128U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 8U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 8U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 8U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 8U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (16U * (threadIdx.x / 32U) * 32U + dotIdx * 16U), 16U);
            wmma::load_matrix_sync(bFrag, sarB + 8U * __anf02 * 16U, 8U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 8U)) * 32U +
                                     blockIdx.x % (cols / 8U) * 8U +
                                     cols * (threadIdx.x / 32U) * 32U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_8x32x16_8x32x16
*/
static void
__hoisted_g_gemm_f16_f16_8x32x16_8x32x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 8U, 32U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 8U, 32U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 8U, 32U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                 blockIdx.x % (cols / 32U) * 32U +
                 cols * (threadIdx.x / 32U) * 8U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 32U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 32U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (32U * (threadIdx.x / 32U) * 8U + dotIdx * 16U), 32U);
            wmma::load_matrix_sync(bFrag, sarB + 32U * __anf02 * 16U, 32U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                                     blockIdx.x % (cols / 32U) * 32U +
                                     cols * (threadIdx.x / 32U) * 8U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_16x16x16
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                 blockIdx.x % (cols / 64U) * 64U +
                 cols * (threadIdx.x / 32U / 4U) * 16U +
                 threadIdx.x / 32U % 4U * 16U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (64U * (threadIdx.x / 32U / 4U) * 16U + dotIdx * 16U),
                64U);
            wmma::load_matrix_sync(bFrag,
                sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 4U * 16U),
                64U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                                     blockIdx.x % (cols / 64U) * 64U +
                                     cols * (threadIdx.x / 32U / 4U) * 16U +
                                     threadIdx.x / 32U % 4U * 16U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_32x8x16
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_32x8x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 32U, 8U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 32U, 8U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 32U, 8U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                 blockIdx.x % (cols / 64U) * 64U +
                 cols * (threadIdx.x / 32U / 8U) * 32U +
                 threadIdx.x / 32U % 8U * 8U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (64U * (threadIdx.x / 32U / 8U) * 32U + dotIdx * 16U),
                64U);
            wmma::load_matrix_sync(bFrag,
                sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 8U * 8U),
                64U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                                     blockIdx.x % (cols / 64U) * 64U +
                                     cols * (threadIdx.x / 32U / 8U) * 32U +
                                     threadIdx.x / 32U % 8U * 8U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_64x64x64_8x32x16
*/
static void
__hoisted_g_gemm_f16_f16_64x64x64_8x32x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 64U;
    uint32_t num_n_tiles = cols / 64U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 8U, 32U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 8U, 32U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 8U, 32U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                 blockIdx.x % (cols / 64U) * 64U +
                 cols * (threadIdx.x / 32U / 2U) * 8U +
                 threadIdx.x / 32U % 2U * 32U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 4096U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 64U;
            uint32_t col = (i + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gA + (shared * mrow * 64U + __anf0 * 64U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 64U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 4096U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 64U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 64U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 64U + mcol * 64U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 64U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 4U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (64U * (threadIdx.x / 32U / 2U) * 8U + dotIdx * 16U),
                64U);
            wmma::load_matrix_sync(bFrag,
                sarB + (64U * __anf02 * 16U + threadIdx.x / 32U % 2U * 32U),
                64U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 64U)) * 64U +
                                     blockIdx.x % (cols / 64U) * 64U +
                                     cols * (threadIdx.x / 32U / 2U) * 8U +
                                     threadIdx.x / 32U % 2U * 32U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_32x32x32_16x16x16
*/
static void
__hoisted_g_gemm_f16_f16_32x32x32_16x16x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                 blockIdx.x % (cols / 32U) * 32U +
                 cols * (threadIdx.x / 32U / 2U) * 16U +
                 threadIdx.x / 32U % 2U * 16U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 32U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 32U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (32U * (threadIdx.x / 32U / 2U) * 16U + dotIdx * 16U),
                32U);
            wmma::load_matrix_sync(bFrag,
                sarB + (32U * __anf02 * 16U + threadIdx.x / 32U % 2U * 16U),
                32U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                                     blockIdx.x % (cols / 32U) * 32U +
                                     cols * (threadIdx.x / 32U / 2U) * 16U +
                                     threadIdx.x / 32U % 2U * 16U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f16_16x16x16_16x16x16
*/
static void
__hoisted_g_gemm_f16_f16_16x16x16_16x16x16_0(
    uint32_t shared, uint32_t cols, half *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(512U);
    uint32_t num_k_tiles = shared / 16U;
    uint32_t num_n_tiles = cols / 16U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 16U)) * 16U +
                 blockIdx.x % (cols / 16U) * 16U +
                 cols * (threadIdx.x / 32U) * 16U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 256U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 16U;
            uint32_t col = (i + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gA + (shared * mrow * 16U + __anf0 * 16U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 16U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 256U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 16U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 16U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 16U + mcol * 16U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 16U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 1U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (16U * (threadIdx.x / 32U) * 16U + dotIdx * 16U), 16U);
            wmma::load_matrix_sync(bFrag, sarB + 16U * __anf02 * 16U, 16U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 16U)) * 16U +
                                     blockIdx.x % (cols / 16U) * 16U +
                                     cols * (threadIdx.x / 32U) * 16U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

__global__
/**
  hoisted when extracting g_gemm_f16_f32_32x32x32_16x16x16
*/
static void
__hoisted_g_gemm_f16_f32_32x32x32_16x16x16_0(uint32_t shared, uint32_t cols,
    float *gC, half *gA, uint32_t nthr, half *gB)
{
    half *sarA = (half *) KPR_SHMEM_AT(0U);
    half *sarB = (half *) KPR_SHMEM_AT(2048U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 32U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &aFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major));
    auto &bFrag = KPR_INIT(
        kpr_fragment(wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major));
    auto &accumFrag =
        KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float));
    wmma::load_matrix_sync(accumFrag,
        gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                 blockIdx.x % (cols / 32U) * 32U +
                 cols * (threadIdx.x / 32U / 2U) * 16U +
                 threadIdx.x / 32U % 2U * 16U),
        cols, wmma::mem_row_major);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf0 = bkIdx;
        uint32_t i = 0U;
        for (; i < 1024U; i += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 32U;
            uint32_t col = (i + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gA + (shared * mrow * 32U + __anf0 * 32U + shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarA[row * 32U + col + k] = local[k];
        }
        uint32_t i1 = 0U;
        for (; i1 < 1024U; i1 += nthr * 8U) {
            KRML_CHECK_SIZE(sizeof(half), 8U);
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i1 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i1 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                gB + (cols * __anf0 * 32U + mcol * 32U + cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sarB[row * 32U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf02 = dotIdx;
            wmma::load_matrix_sync(aFrag,
                sarA + (32U * (threadIdx.x / 32U / 2U) * 16U + dotIdx * 16U),
                32U);
            wmma::load_matrix_sync(bFrag,
                sarB + (32U * __anf02 * 16U + threadIdx.x / 32U % 2U * 16U),
                32U);
            wmma::mma_sync(accumFrag, aFrag, bFrag, accumFrag);
        }
    }
    wmma::store_matrix_sync(gC + (cols * (blockIdx.x / (cols / 32U)) * 32U +
                                     blockIdx.x % (cols / 32U) * 32U +
                                     cols * (threadIdx.x / 32U / 2U) * 16U +
                                     threadIdx.x / 32U % 2U * 16U),
        accumFrag, cols, wmma::mem_row_major);
    __syncwarp();
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x16_16x16x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x16_16x16x16_0, nblk, 512U, 4096U,
        s, shared, cols, gC, gA, 512U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_32x8x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t nblk = rows / 32U * (cols / 32U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_32x32x32_32x8x16_0, nblk, 128U, 4096U, s,
        shared, cols, gC, gA, 128U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_8x32x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t nblk = rows / 32U * (cols / 32U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_32x32x32_8x32x16_0, nblk, 128U, 4096U, s,
        shared, cols, gC, gA, 128U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_32x8x16_32x8x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 8U == 0U);
    uint32_t nblk = rows / 32U * (cols / 8U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(1280U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_32x8x16_32x8x16_0, nblk, 32U, 1280U, s,
        shared, cols, gC, gA, 32U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_8x32x16_8x32x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t nblk = rows / 32U * (cols / 32U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_8x32x16_8x32x16_0, nblk, 128U, 4096U, s,
        shared, cols, gC, gA, 128U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_16x16x16_0, nblk, 512U, 16384U,
        s, shared, cols, gC, gA, 512U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_32x8x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_32x8x16_0, nblk, 512U, 16384U,
        s, shared, cols, gC, gA, 512U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_8x32x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 64U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    uint32_t nblk = rows / 64U * (cols / 64U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_64x64x64_8x32x16_0, nblk, 512U, 16384U,
        s, shared, cols, gC, gA, 512U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_16x16x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t nblk = rows / 32U * (cols / 32U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_32x32x32_16x16x16_0, nblk, 128U, 4096U,
        s, shared, cols, gC, gA, 128U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f16_16x16x16_16x16x16(
    uint32_t rows, uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t nblk = rows / 16U * (cols / 16U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(1024U);
    KPR_KCALL(__hoisted_g_gemm_f16_f16_16x16x16_16x16x16_0, nblk, 32U, 1024U, s,
        shared, cols, gC, gA, 32U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_TensorCore_g_gemm_f16_f32_32x32x32_16x16x16(uint32_t rows,
    uint32_t shared, uint32_t cols, half *gA, half *gB, float *gC)
{
    KPR_GUARD(rows * cols <= 2097152U);
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t nblk = rows / 32U * (cols / 32U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f16_f32_32x32x32_16x16x16_0, nblk, 128U, 4096U,
        s, shared, cols, gC, gA, 128U, gB);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
