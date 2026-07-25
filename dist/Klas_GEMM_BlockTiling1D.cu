
#include "Klas_GEMM_BlockTiling1D.h"

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
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float *gA_p = gA;
    float *gB_p = gB;
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(blockIdx.x / nn * 32U + row) * n + blockIdx.x % nn * 32U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_matmul_f32_tile32_rrr_0,
              mm * nn, 32U, 8192U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double *gA_p = gA;
    double *gB_p = gB;
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(blockIdx.x / nn * 32U + row) * n + blockIdx.x % nn * 32U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                double *gA,
                                                double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_matmul_f64_tile32_rrr_0,
              mm * nn, 32U, 16384U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t *gA_p = gA;
    uint32_t *gB_p = gB;
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(blockIdx.x / nn * 32U + row) * n + blockIdx.x % nn * 32U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_matmul_u32_tile32_rrr_0,
              mm * nn, 32U, 8192U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t *gA_p = gA;
    uint64_t *gB_p = gB;
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(blockIdx.x / nn * 32U + row) * n + blockIdx.x % nn * 32U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_matmul_u64_tile32_rrr_0,
              mm * nn, 32U, 16384U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float *gA_p = gA;
    float *gB_p = gB;
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(blockIdx.x / nn * 16U + row) * n + blockIdx.x % nn * 16U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_matmul_f32_tile16_rrr_0,
              mm * nn, 16U, 2048U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double *gA_p = gA;
    double *gB_p = gB;
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(blockIdx.x / nn * 16U + row) * n + blockIdx.x % nn * 16U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                double *gA,
                                                double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_matmul_f64_tile16_rrr_0,
              mm * nn, 16U, 4096U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t *gA_p = gA;
    uint32_t *gB_p = gB;
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(blockIdx.x / nn * 16U + row) * n + blockIdx.x % nn * 16U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_matmul_u32_tile16_rrr_0,
              mm * nn, 16U, 2048U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t *gA_p = gA;
    uint64_t *gB_p = gB;
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(blockIdx.x / nn * 16U + row) * n + blockIdx.x % nn * 16U +
           threadIdx.x] = sums[row];
}

void
Klas_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_matmul_u64_tile16_rrr_0,
              mm * nn, 16U, 4096U, s, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float *gA_p = gA;
    float *gB_p = gB;
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 32U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(float alpha,
                                              float beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_tile32_rrr_0,
              mm * nn, 32U, 8192U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double *gA_p = gA;
    double *gB_p = gB;
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 32U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(double alpha,
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f64_tile32_rrr_0,
              mm * nn, 32U, 16384U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t *gA_p = gA;
    uint32_t *gB_p = gB;
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 32U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(uint32_t alpha,
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_u32_tile32_rrr_0,
              mm * nn, 32U, 8192U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t *gA_p = gA;
    uint64_t *gB_p = gB;
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 32U + threadIdx.x] =
                gA_p[(32U * (blockIdx.x / nn) + ci) * k + 32U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 32U + threadIdx.x] =
                gB_p[(32U * __anf06 + ci1) * n + 32U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 32U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(uint64_t alpha,
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_u64_tile32_rrr_0,
              mm * nn, 32U, 16384U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float *gA_p = gA;
    float *gB_p = gB;
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 16U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(float alpha,
                                              float beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_tile16_rrr_0,
              mm * nn, 16U, 2048U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double *gA_p = gA;
    double *gB_p = gB;
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 16U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(double alpha,
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_f64_tile16_rrr_0,
              mm * nn, 16U, 4096U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t *gA_p = gA;
    uint32_t *gB_p = gB;
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 16U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(uint32_t alpha,
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_gemm_u32_tile16_rrr_0,
              mm * nn, 16U, 2048U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
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
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t *gA_p = gA;
    uint64_t *gB_p = gB;
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf06 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t ci = i0;
            sa1[ci * 16U + threadIdx.x] =
                gA_p[(16U * (blockIdx.x / nn) + ci) * k + 16U * __anf06 +
                     threadIdx.x];
            uint32_t ci1 = i0;
            sa2[ci1 * 16U + threadIdx.x] =
                gB_p[(16U * __anf06 + ci1) * n + 16U * (blockIdx.x % nn) +
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
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = blockIdx.x / nn * 16U + row;
        uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void
Klas_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(uint64_t alpha,
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
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_u64_tile16_rrr_0,
              mm * nn, 16U, 4096U, s, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
