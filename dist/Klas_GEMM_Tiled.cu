
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *abibk = gA;
        uint32_t __anf01 = bk;
        float *bbkbj = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        float s_ = sum1;
        sum += s_;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f32_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0,
              m / tile * nn,
              tile * tile, 0U, tile, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *abibk = gA;
        uint32_t __anf01 = bk;
        double *bbkbj = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        double s_ = sum1;
        sum += s_;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

void
Klas_GEMM_Tiled_g_matmul_f64_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0,
              m / tile * nn,
              tile * tile, 0U, tile, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint32_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        uint32_t s_ = sum1;
        sum += s_;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u32_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_matmul_u32_rrr_0,
              m / tile * nn,
              tile * tile, 0U, tile, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint64_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        uint64_t s_ = sum1;
        sum += s_;
    }
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] = sum;
}

void
Klas_GEMM_Tiled_g_matmul_u64_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_matmul_u64_rrr_0,
              m / tile * nn,
              tile * tile, 0U, tile, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *abibk = gA;
        uint32_t __anf01 = bk;
        float *bbkbj = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        float s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *abibk = gA;
        uint32_t __anf01 = bk;
        double *bbkbj = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        double s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint32_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        uint32_t s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint64_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        uint64_t s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *abibk = gA;
        uint32_t __anf01 = bk;
        float *bbkbj = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        float s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *abibk = gA;
        uint32_t __anf01 = bk;
        double *bbkbj = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        double s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint32_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        uint32_t s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint64_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        uint64_t s_ = sum1;
        sum += s_;
    }
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        sum;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *abibk = gA;
        uint32_t __anf01 = bk;
        float *bbkbj = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        float s_ = sum1;
        sum += s_;
    }
    float s = sum;
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f32_rrr(uint32_t tile,
                               float alpha,
                               float beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_gemm_f32_rrr_0,
              m / tile * nn,
              tile * tile,
              0U, tile, alpha, beta, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *abibk = gA;
        uint32_t __anf01 = bk;
        double *bbkbj = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        double s_ = sum1;
        sum += s_;
    }
    double s = sum;
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * s;
}

void
Klas_GEMM_Tiled_g_gemm_f64_rrr(uint32_t tile,
                               double alpha,
                               double beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_gemm_f64_rrr_0,
              m / tile * nn,
              tile * tile,
              0U, tile, alpha, beta, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint32_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        uint32_t s_ = sum1;
        sum += s_;
    }
    uint32_t s = sum;
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * s;
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
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_gemm_u32_rrr_0,
              m / tile * nn,
              tile * tile,
              0U, tile, alpha, beta, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t brow = threadIdx.x / tile;
    uint32_t bcol = threadIdx.x % tile;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint64_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * tile + brow) * k + __anf0 * tile + vk] *
                bbkbj[(__anf01 * tile + vk) * n + mcol * tile + bcol];
        }
        uint64_t s_ = sum1;
        sum += s_;
    }
    uint64_t s = sum;
    gC[(mrow * tile + brow) * n + mcol * tile + bcol] =
        beta * gC[(mrow * tile + brow) * n + mcol * tile + bcol] + alpha * s;
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
    uint32_t nn = n / tile;
    KPR_KCALL(__hoisted_g_gemm_u64_rrr_0,
              m / tile * nn,
              tile * tile,
              0U, tile, alpha, beta, n, k, gA, gB, gC, nn, k / tile);
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *abibk = gA;
        uint32_t __anf01 = bk;
        float *bbkbj = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        float s_ = sum1;
        sum += s_;
    }
    float s = sum;
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                  threadIdx.x % 32U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *abibk = gA;
        uint32_t __anf01 = bk;
        double *bbkbj = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        double s_ = sum1;
        sum += s_;
    }
    double s = sum;
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                  threadIdx.x % 32U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint32_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        uint32_t s_ = sum1;
        sum += s_;
    }
    uint32_t s = sum;
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                  threadIdx.x % 32U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint64_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 32U + threadIdx.x / 32U) * k + __anf0 * 32U +
                      vk] * bbkbj[(__anf01 * 32U + vk) * n + mcol * 32U +
                                  threadIdx.x % 32U];
        }
        uint64_t s_ = sum1;
        sum += s_;
    }
    uint64_t s = sum;
    gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U + threadIdx.x % 32U] =
        beta * gC[(mrow * 32U + threadIdx.x / 32U) * n + mcol * 32U +
                  threadIdx.x % 32U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        float *abibk = gA;
        uint32_t __anf01 = bk;
        float *bbkbj = gB;
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        float s_ = sum1;
        sum += s_;
    }
    float s = sum;
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                  threadIdx.x % 16U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        double *abibk = gA;
        uint32_t __anf01 = bk;
        double *bbkbj = gB;
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        double s_ = sum1;
        sum += s_;
    }
    double s = sum;
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                  threadIdx.x % 16U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint32_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint32_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        uint32_t s_ = sum1;
        sum += s_;
    }
    uint32_t s = sum;
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                  threadIdx.x % 16U] + alpha * s;
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
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t __anf0 = bk;
        uint64_t *abibk = gA;
        uint32_t __anf01 = bk;
        uint64_t *bbkbj = gB;
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                abibk[(mrow * 16U + threadIdx.x / 16U) * k + __anf0 * 16U +
                      vk] * bbkbj[(__anf01 * 16U + vk) * n + mcol * 16U +
                                  threadIdx.x % 16U];
        }
        uint64_t s_ = sum1;
        sum += s_;
    }
    uint64_t s = sum;
    gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U + threadIdx.x % 16U] =
        beta * gC[(mrow * 16U + threadIdx.x / 16U) * n + mcol * 16U +
                  threadIdx.x % 16U] + alpha * s;
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
