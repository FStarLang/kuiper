
#include "Klas_GEMM_Tiled.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_g_matmul_f32_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk, float *gA,
    float *gB, uint32_t k, uint32_t n, float *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_g_matmul_f64_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk,
    double *gA, double *gB, uint32_t k, uint32_t n, double *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_g_matmul_u32_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk,
    uint32_t *gA, uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_g_matmul_u64_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk,
    uint64_t *gA, uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_g_matmul_f32_tile32_rrr_0(uint32_t nn, uint32_t kk, float *gA,
    float *gB, uint32_t k, uint32_t n, float *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_g_matmul_f64_tile32_rrr_0(uint32_t nn, uint32_t kk, double *gA,
    double *gB, uint32_t k, uint32_t n, double *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_g_matmul_u32_tile32_rrr_0(uint32_t nn, uint32_t kk, uint32_t *gA,
    uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_g_matmul_u64_tile32_rrr_0(uint32_t nn, uint32_t kk, uint64_t *gA,
    uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_g_matmul_f32_tile16_rrr_0(uint32_t nn, uint32_t kk, float *gA,
    float *gB, uint32_t k, uint32_t n, float *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_g_matmul_f64_tile16_rrr_0(uint32_t nn, uint32_t kk, double *gA,
    double *gB, uint32_t k, uint32_t n, double *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_g_matmul_u32_tile16_rrr_0(uint32_t nn, uint32_t kk, uint32_t *gA,
    uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_g_matmul_u64_tile16_rrr_0(uint32_t nn, uint32_t kk, uint64_t *gA,
    uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
}

__global__
/**
  hoisted when extracting g_gemm_f32_rrr
*/
static void
__hoisted_g_gemm_f32_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk, float *gA,
    float *gB, uint32_t k, uint32_t n, float *gC, float beta, float alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * sum;
}

__global__
/**
  hoisted when extracting g_gemm_f64_rrr
*/
static void
__hoisted_g_gemm_f64_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk, double *gA,
    double *gB, uint32_t k, uint32_t n, double *gC, double beta, double alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * sum;
}

__global__
/**
  hoisted when extracting g_gemm_u32_rrr
*/
static void
__hoisted_g_gemm_u32_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk,
    uint32_t *gA, uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC,
    uint32_t beta, uint32_t alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * sum;
}

__global__
/**
  hoisted when extracting g_gemm_u64_rrr
*/
static void
__hoisted_g_gemm_u64_rrr_0(uint32_t nn, uint32_t tile, uint32_t kk,
    uint64_t *gA, uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC,
    uint64_t beta, uint64_t alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += gA[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                    gB[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        sum += sum1;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_f32_tile32_rrr
*/
static void
__hoisted_g_gemm_f32_tile32_rrr_0(uint32_t nn, uint32_t kk, float *gA,
    float *gB, uint32_t k, uint32_t n, float *gC, float beta, float alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                   threadIdx.x % 32U] +
        alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_f64_tile32_rrr
*/
static void
__hoisted_g_gemm_f64_tile32_rrr_0(uint32_t nn, uint32_t kk, double *gA,
    double *gB, uint32_t k, uint32_t n, double *gC, double beta, double alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                   threadIdx.x % 32U] +
        alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_u32_tile32_rrr
*/
static void
__hoisted_g_gemm_u32_tile32_rrr_0(uint32_t nn, uint32_t kk, uint32_t *gA,
    uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC, uint32_t beta,
    uint32_t alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                   threadIdx.x % 32U] +
        alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_u64_tile32_rrr
*/
static void
__hoisted_g_gemm_u64_tile32_rrr_0(uint32_t nn, uint32_t kk, uint64_t *gA,
    uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC, uint64_t beta,
    uint64_t alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U + vk] *
                gB[(__anf01 * 32U + vk) * n + mcol * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                   threadIdx.x % 32U] +
        alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_f32_tile16_rrr
*/
static void
__hoisted_g_gemm_f32_tile16_rrr_0(uint32_t nn, uint32_t kk, float *gA,
    float *gB, uint32_t k, uint32_t n, float *gC, float beta, float alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                   threadIdx.x % 16U] +
        alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_f64_tile16_rrr
*/
static void
__hoisted_g_gemm_f64_tile16_rrr_0(uint32_t nn, uint32_t kk, double *gA,
    double *gB, uint32_t k, uint32_t n, double *gC, double beta, double alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                   threadIdx.x % 16U] +
        alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_u32_tile16_rrr
*/
static void
__hoisted_g_gemm_u32_tile16_rrr_0(uint32_t nn, uint32_t kk, uint32_t *gA,
    uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC, uint32_t beta,
    uint32_t alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                   threadIdx.x % 16U] +
        alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_u64_tile16_rrr
*/
static void
__hoisted_g_gemm_u64_tile16_rrr_0(uint32_t nn, uint32_t kk, uint64_t *gA,
    uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC, uint64_t beta,
    uint64_t alpha)
{
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t __anf01 = bk;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                gA[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U + vk] *
                gB[(__anf01 * 16U + vk) * n + mcol * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                   threadIdx.x % 16U] +
        alpha * sum;
}

void Klas_GEMM_Tiled_g_matmul_f32_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0, mm * nn, tile * tile, 0U, s, nn,
        tile, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_f64_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0, mm * nn, tile * tile, 0U, s, nn,
        tile, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_u32_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_u32_rrr_0, mm * nn, tile * tile, 0U, s, nn,
        tile, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_u64_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_u64_rrr_0, mm * nn, tile * tile, 0U, s, nn,
        tile, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_f32_tile32_rrr(
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_f32_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn,
        kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_f64_tile32_rrr(
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_f64_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn,
        kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_u32_tile32_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_u32_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn,
        kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_u64_tile32_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_u64_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn,
        kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_f32_tile16_rrr(
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_f32_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_f64_tile16_rrr(
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_f64_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_u32_tile16_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_u32_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_matmul_u64_tile16_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_matmul_u64_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_f32_rrr(uint32_t tile, float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_f32_rrr_0, mm * nn, tile * tile, 0U, s, nn, tile,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_f64_rrr(uint32_t tile, double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_f64_rrr_0, mm * nn, tile * tile, 0U, s, nn, tile,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_u32_rrr(uint32_t tile, uint32_t alpha,
    uint32_t beta, uint32_t m, uint32_t n, uint32_t k, uint32_t *gA,
    uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_u32_rrr_0, mm * nn, tile * tile, 0U, s, nn, tile,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_u64_rrr(uint32_t tile, uint64_t alpha,
    uint64_t beta, uint32_t m, uint32_t n, uint32_t k, uint64_t *gA,
    uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_u64_rrr_0, mm * nn, tile * tile, 0U, s, nn, tile,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_f32_tile32_rrr(float alpha, float beta, uint32_t m,
    uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_f32_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_f64_tile32_rrr(double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_f64_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_u32_tile32_rrr(uint32_t alpha, uint32_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
    uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_u32_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_u64_tile32_rrr(uint64_t alpha, uint64_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
    uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_u64_tile32_rrr_0, mm * nn, 1024U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_f32_tile16_rrr(float alpha, float beta, uint32_t m,
    uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_f32_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_f64_tile16_rrr(double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_f64_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_u32_tile16_rrr(uint32_t alpha, uint32_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
    uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_u32_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_Tiled_g_gemm_u64_tile16_rrr(uint64_t alpha, uint64_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
    uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_g_gemm_u64_tile16_rrr_0, mm * nn, 256U, 0U, s, nn, kk,
        gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
