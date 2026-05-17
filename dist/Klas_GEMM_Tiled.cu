
#include "Klas_GEMM_Tiled.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_g_matmul_f32_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             float *gA,
                             float *gB, float *gC, uint32_t nn, uint32_t kk)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *tA = gA;
        uint32_t __anf01 = bk;
        float *tB = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f32_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0,
              mm * nn, tile * tile, 0U, tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_g_matmul_f64_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             double *gA,
                             double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *tA = gA;
        uint32_t __anf01 = bk;
        double *tB = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f64_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0,
              mm * nn, tile * tile, 0U, tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_g_matmul_u32_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             uint32_t *gA,
                             uint32_t *gB,
                             uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *tA = gA;
        uint32_t __anf01 = bk;
        uint32_t *tB = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u32_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_matmul_u32_rrr_0,
              mm * nn, tile * tile, 0U, tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_g_matmul_u64_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             uint64_t *gA,
                             uint64_t *gB,
                             uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *tA = gA;
        uint32_t __anf01 = bk;
        uint64_t *tB = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u64_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_matmul_u64_rrr_0,
              mm * nn, tile * tile, 0U, tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_g_matmul_f32_tile32_rrr_0(uint32_t n,
                                    uint32_t k,
                                    float *gA,
                                    float *gB,
                                    float *gC, uint32_t nn, uint32_t kk)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *tA = gA;
        uint32_t __anf01 = bk;
        float *tB = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f32_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_matmul_f32_tile32_rrr_0,
              m / 32U * nn, 1024U, 0U, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_g_matmul_f64_tile32_rrr_0(uint32_t n,
                                    uint32_t k,
                                    double *gA,
                                    double *gB,
                                    double *gC, uint32_t nn, uint32_t kk)
{
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *tA = gA;
        uint32_t __anf01 = bk;
        double *tB = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f64_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        double *gA, double *gB, double *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_matmul_f64_tile32_rrr_0,
              m / 32U * nn, 1024U, 0U, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_g_matmul_u32_tile32_rrr_0(uint32_t n,
                                    uint32_t k,
                                    uint32_t *gA,
                                    uint32_t *gB,
                                    uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *tA = gA;
        uint32_t __anf01 = bk;
        uint32_t *tB = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u32_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t *gA,
                                        uint32_t *gB, uint32_t *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_matmul_u32_tile32_rrr_0,
              m / 32U * nn, 1024U, 0U, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_g_matmul_u64_tile32_rrr_0(uint32_t n,
                                    uint32_t k,
                                    uint64_t *gA,
                                    uint64_t *gB,
                                    uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *tA = gA;
        uint32_t __anf01 = bk;
        uint64_t *tB = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u64_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint64_t *gA,
                                        uint64_t *gB, uint64_t *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_matmul_u64_tile32_rrr_0,
              m / 32U * nn, 1024U, 0U, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_g_matmul_f32_tile16_rrr_0(uint32_t n,
                                    uint32_t k,
                                    float *gA,
                                    float *gB,
                                    float *gC, uint32_t nn, uint32_t kk)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *tA = gA;
        uint32_t __anf01 = bk;
        float *tB = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f32_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_matmul_f32_tile16_rrr_0,
              m / 16U * nn, 256U, 0U, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_g_matmul_f64_tile16_rrr_0(uint32_t n,
                                    uint32_t k,
                                    double *gA,
                                    double *gB,
                                    double *gC, uint32_t nn, uint32_t kk)
{
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *tA = gA;
        uint32_t __anf01 = bk;
        double *tB = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f64_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        double *gA, double *gB, double *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_matmul_f64_tile16_rrr_0,
              m / 16U * nn, 256U, 0U, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_g_matmul_u32_tile16_rrr_0(uint32_t n,
                                    uint32_t k,
                                    uint32_t *gA,
                                    uint32_t *gB,
                                    uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *tA = gA;
        uint32_t __anf01 = bk;
        uint32_t *tB = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u32_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t *gA,
                                        uint32_t *gB, uint32_t *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_matmul_u32_tile16_rrr_0,
              m / 16U * nn, 256U, 0U, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_g_matmul_u64_tile16_rrr_0(uint32_t n,
                                    uint32_t k,
                                    uint64_t *gA,
                                    uint64_t *gB,
                                    uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *tA = gA;
        uint32_t __anf01 = bk;
        uint64_t *tB = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u64_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint64_t *gA,
                                        uint64_t *gB, uint64_t *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_matmul_u64_tile16_rrr_0,
              m / 16U * nn, 256U, 0U, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_rrr
*/
static void
__hoisted_g_gemm_f32_rrr_0(uint32_t tile,
                           float alpha,
                           float beta,
                           uint32_t n,
                           uint32_t k,
                           float *gA,
                           float *gB, float *gC, uint32_t nn, uint32_t kk)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *tA = gA;
        uint32_t __anf01 = bk;
        float *tB = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        float s_ = sum1;
        sum += s_;
    }
    float s = sum;
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        =
        beta *
        gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
              tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f32_rrr(uint32_t tile,
                               float alpha,
                               float beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_gemm_f32_rrr_0,
              mm * nn,
              tile * tile, 0U, tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_rrr
*/
static void
__hoisted_g_gemm_f64_rrr_0(uint32_t tile,
                           double alpha,
                           double beta,
                           uint32_t n,
                           uint32_t k,
                           double *gA,
                           double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *tA = gA;
        uint32_t __anf01 = bk;
        double *tB = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        double s_ = sum1;
        sum += s_;
    }
    double s = sum;
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        =
        beta *
        gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
              tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f64_rrr(uint32_t tile,
                               double alpha,
                               double beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_gemm_f64_rrr_0,
              mm * nn,
              tile * tile, 0U, tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_rrr
*/
static void
__hoisted_g_gemm_u32_rrr_0(uint32_t tile,
                           uint32_t alpha,
                           uint32_t beta,
                           uint32_t n,
                           uint32_t k,
                           uint32_t *gA,
                           uint32_t *gB, uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *tA = gA;
        uint32_t __anf01 = bk;
        uint32_t *tB = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        uint32_t s_ = sum1;
        sum += s_;
    }
    uint32_t s = sum;
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        =
        beta *
        gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
              tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_u32_rrr(uint32_t tile,
                               uint32_t alpha,
                               uint32_t beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k,
                               uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_gemm_u32_rrr_0,
              mm * nn,
              tile * tile, 0U, tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_rrr
*/
static void
__hoisted_g_gemm_u64_rrr_0(uint32_t tile,
                           uint64_t alpha,
                           uint64_t beta,
                           uint32_t n,
                           uint32_t k,
                           uint64_t *gA,
                           uint64_t *gB, uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *tA = gA;
        uint32_t __anf01 = bk;
        uint64_t *tB = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++)
            sum1 +=
                tA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                   tile * __anf0 + k1] * tB[(tile * __anf01 + k1) * n +
                                            tile * (blockIdx.x % nn) +
                                            threadIdx.x % tile];
        uint64_t s_ = sum1;
        sum += s_;
    }
    uint64_t s = sum;
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        =
        beta *
        gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
              tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_u64_rrr(uint32_t tile,
                               uint64_t alpha,
                               uint64_t beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k,
                               uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_KCALL(__hoisted_g_gemm_u64_rrr_0,
              mm * nn,
              tile * tile, 0U, tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile32_rrr
*/
static void
__hoisted_g_gemm_f32_tile32_rrr_0(float alpha,
                                  float beta,
                                  uint32_t n,
                                  uint32_t k,
                                  float *gA,
                                  float *gB,
                                  float *gC, uint32_t nn, uint32_t kk)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *tA = gA;
        uint32_t __anf01 = bk;
        float *tB = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        float s_ = sum1;
        sum += s_;
    }
    float s = sum;
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        =
        beta *
        gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
              32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f32_tile32_rrr(float alpha,
                                      float beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_gemm_f32_tile32_rrr_0,
              m / 32U * nn,
              1024U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile32_rrr
*/
static void
__hoisted_g_gemm_f64_tile32_rrr_0(double alpha,
                                  double beta,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA,
                                  double *gB,
                                  double *gC, uint32_t nn, uint32_t kk)
{
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *tA = gA;
        uint32_t __anf01 = bk;
        double *tB = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        double s_ = sum1;
        sum += s_;
    }
    double s = sum;
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        =
        beta *
        gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
              32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f64_tile32_rrr(double alpha,
                                      double beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      double *gA, double *gB, double *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_gemm_f64_tile32_rrr_0,
              m / 32U * nn,
              1024U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile32_rrr
*/
static void
__hoisted_g_gemm_u32_tile32_rrr_0(uint32_t alpha,
                                  uint32_t beta,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t *gA,
                                  uint32_t *gB,
                                  uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *tA = gA;
        uint32_t __anf01 = bk;
        uint32_t *tB = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    uint32_t s = sum;
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        =
        beta *
        gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
              32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                      uint32_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_gemm_u32_tile32_rrr_0,
              m / 32U * nn,
              1024U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile32_rrr
*/
static void
__hoisted_g_gemm_u64_tile32_rrr_0(uint64_t alpha,
                                  uint64_t beta,
                                  uint32_t n,
                                  uint32_t k,
                                  uint64_t *gA,
                                  uint64_t *gB,
                                  uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *tA = gA;
        uint32_t __anf01 = bk;
        uint64_t *tB = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++)
            sum1 +=
                tA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                   32U * __anf0 + k1] * tB[(32U * __anf01 + k1) * n +
                                           32U * (blockIdx.x % nn) +
                                           threadIdx.x % 32U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    uint64_t s = sum;
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        =
        beta *
        gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
              32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                      uint64_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t nn = n / 32U;
    KPR_KCALL(__hoisted_g_gemm_u64_tile32_rrr_0,
              m / 32U * nn,
              1024U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 32U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile16_rrr
*/
static void
__hoisted_g_gemm_f32_tile16_rrr_0(float alpha,
                                  float beta,
                                  uint32_t n,
                                  uint32_t k,
                                  float *gA,
                                  float *gB,
                                  float *gC, uint32_t nn, uint32_t kk)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *tA = gA;
        uint32_t __anf01 = bk;
        float *tB = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        float s_ = sum1;
        sum += s_;
    }
    float s = sum;
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        =
        beta *
        gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
              16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f32_tile16_rrr(float alpha,
                                      float beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_gemm_f32_tile16_rrr_0,
              m / 16U * nn,
              256U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile16_rrr
*/
static void
__hoisted_g_gemm_f64_tile16_rrr_0(double alpha,
                                  double beta,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA,
                                  double *gB,
                                  double *gC, uint32_t nn, uint32_t kk)
{
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *tA = gA;
        uint32_t __anf01 = bk;
        double *tB = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        double s_ = sum1;
        sum += s_;
    }
    double s = sum;
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        =
        beta *
        gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
              16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f64_tile16_rrr(double alpha,
                                      double beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      double *gA, double *gB, double *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_gemm_f64_tile16_rrr_0,
              m / 16U * nn,
              256U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile16_rrr
*/
static void
__hoisted_g_gemm_u32_tile16_rrr_0(uint32_t alpha,
                                  uint32_t beta,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t *gA,
                                  uint32_t *gB,
                                  uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *tA = gA;
        uint32_t __anf01 = bk;
        uint32_t *tB = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    uint32_t s = sum;
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        =
        beta *
        gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
              16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                      uint32_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_gemm_u32_tile16_rrr_0,
              m / 16U * nn,
              256U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile16_rrr
*/
static void
__hoisted_g_gemm_u64_tile16_rrr_0(uint64_t alpha,
                                  uint64_t beta,
                                  uint32_t n,
                                  uint32_t k,
                                  uint64_t *gA,
                                  uint64_t *gB,
                                  uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *tA = gA;
        uint32_t __anf01 = bk;
        uint64_t *tB = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++)
            sum1 +=
                tA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                   16U * __anf0 + k1] * tB[(16U * __anf01 + k1) * n +
                                           16U * (blockIdx.x % nn) +
                                           threadIdx.x % 16U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    uint64_t s = sum;
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        =
        beta *
        gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
              16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                      uint64_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t nn = n / 16U;
    KPR_KCALL(__hoisted_g_gemm_u64_tile16_rrr_0,
              m / 16U * nn,
              256U, 0U, alpha, beta, n, k, gA, gB, gC, nn, k / 16U);
    MUST(cudaDeviceSynchronize());
}
