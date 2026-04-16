
#include "Kuiper_GEMM_BlockTiling1D.h"

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_0(uint32_t n, uint32_t k, float *gA, float *gB, float *gC,
            uint32_t nn, uint32_t kk)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            float v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    float *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_0, mm * nn, 32U, 8192U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_1(uint32_t n,
            uint32_t k,
            double *gA, double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            double v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    double *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  double *gA,
                                                  double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_1, mm * nn, 32U, 16384U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_2(uint32_t n,
            uint32_t k,
            uint32_t *gA, uint32_t *gB, uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            uint32_t v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    uint32_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  uint32_t *gA,
                                                  uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_2, mm * nn, 32U, 8192U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_3(uint32_t n,
            uint32_t k,
            uint64_t *gA, uint64_t *gB, uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            uint64_t v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    uint64_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  uint64_t *gA,
                                                  uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_3, mm * nn, 32U, 16384U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_4(uint32_t n, uint32_t k, float *gA, float *gB, float *gC,
            uint32_t nn, uint32_t kk)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            float v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    float *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_4, mm * nn, 16U, 2048U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_5(uint32_t n,
            uint32_t k,
            double *gA, double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            double v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    double *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  double *gA,
                                                  double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_5, mm * nn, 16U, 4096U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_6(uint32_t n,
            uint32_t k,
            uint32_t *gA, uint32_t *gB, uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            uint32_t v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    uint32_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  uint32_t *gA,
                                                  uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_6, mm * nn, 16U, 2048U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_7(uint32_t n,
            uint32_t k,
            uint64_t *gA, uint64_t *gB, uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            uint64_t v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    uint64_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(uint32_t m,
                                                  uint32_t n,
                                                  uint32_t k,
                                                  uint64_t *gA,
                                                  uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_7, mm * nn, 16U, 4096U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile32_rrr
*/
static void
__hoisted_8(float alpha,
            float beta,
            uint32_t n,
            uint32_t k,
            float *gA, float *gB, float *gC, uint32_t nn, uint32_t kk)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            float v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    float *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(32U * (blockIdx.x / nn) + row) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_8, mm * nn, 32U, 8192U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile32_rrr
*/
static void
__hoisted_9(double alpha,
            double beta,
            uint32_t n,
            uint32_t k,
            double *gA, double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            double v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    double *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(32U * (blockIdx.x / nn) + row) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(double alpha,
                                                double beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                double *gA,
                                                double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_9, mm * nn, 32U, 16384U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile32_rrr
*/
static void
__hoisted_10(uint32_t alpha,
             uint32_t beta,
             uint32_t n,
             uint32_t k,
             uint32_t *gA, uint32_t *gB, uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            uint32_t v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    uint32_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(32U * (blockIdx.x / nn) + row) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_10, mm * nn, 32U, 8192U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile32_rrr
*/
static void
__hoisted_11(uint64_t alpha,
             uint64_t beta,
             uint32_t n,
             uint32_t k,
             uint64_t *gA, uint64_t *gB, uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            sa1[i0 * 32U + threadIdx.x] =
                gA[(32U * (blockIdx.x / nn) + i0) * k + 32U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + i0) * n + 32U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i = 0U;
            uint64_t v2 = sa2[sk * 32U + threadIdx.x];
            for (; i < 32U; i++)
                sums[i] += sa1[i * 32U + sk] * v2;
        }
    }
    uint64_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 32U; row++)
        tileC[(32U * (blockIdx.x / nn) + row) * n + 32U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(32U * (blockIdx.x / nn) + row) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_11, mm * nn, 32U, 16384U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile16_rrr
*/
static void
__hoisted_12(float alpha,
             float beta,
             uint32_t n,
             uint32_t k,
             float *gA, float *gB, float *gC, uint32_t nn, uint32_t kk)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            float v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    float *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(16U * (blockIdx.x / nn) + row) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(float alpha,
                                                float beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_12, mm * nn, 16U, 2048U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile16_rrr
*/
static void
__hoisted_13(double alpha,
             double beta,
             uint32_t n,
             uint32_t k,
             double *gA, double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            double v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    double *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(16U * (blockIdx.x / nn) + row) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(double alpha,
                                                double beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                double *gA,
                                                double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_13, mm * nn, 16U, 4096U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile16_rrr
*/
static void
__hoisted_14(uint32_t alpha,
             uint32_t beta,
             uint32_t n,
             uint32_t k,
             uint32_t *gA, uint32_t *gB, uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            uint32_t v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    uint32_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(16U * (blockIdx.x / nn) + row) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_14, mm * nn, 16U, 2048U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile16_rrr
*/
static void
__hoisted_15(uint64_t alpha,
             uint64_t beta,
             uint32_t n,
             uint32_t k,
             uint64_t *gA, uint64_t *gB, uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            sa1[i0 * 16U + threadIdx.x] =
                gA[(16U * (blockIdx.x / nn) + i0) * k + 16U * __anf0 +
                   threadIdx.x];
            sa2[i0 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + i0) * n + 16U * (blockIdx.x % nn) +
                   threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i = 0U;
            uint64_t v2 = sa2[sk * 16U + threadIdx.x];
            for (; i < 16U; i++)
                sums[i] += sa1[i * 16U + sk] * v2;
        }
    }
    uint64_t *tileC = gC;
    uint32_t row = 0U;
    for (; row < 16U; row++)
        tileC[(16U * (blockIdx.x / nn) + row) * n + 16U * (blockIdx.x % nn) +
              threadIdx.x] =
            beta * tileC[(16U * (blockIdx.x / nn) + row) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x] +
            alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_15, mm * nn, 16U, 4096U, alpha, beta, n, k, gA, gB, gC,
              nn, kk);
    MUST(cudaDeviceSynchronize());
}
