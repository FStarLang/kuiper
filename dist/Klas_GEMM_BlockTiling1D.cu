
#include "Klas_GEMM_BlockTiling1D.h"

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_g_matmul_f32_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    float *gA, float *gB, uint32_t k, uint32_t n, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT(4096U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            float v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(mrow * 32U + row) * n + mcol * 32U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_g_matmul_f64_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    double *gA, double *gB, uint32_t k, uint32_t n, double *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT(8192U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            double v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(mrow * 32U + row) * n + mcol * 32U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_g_matmul_u32_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint32_t *gA, uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sums[32U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            uint32_t v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(mrow * 32U + row) * n + mcol * 32U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_g_matmul_u64_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint64_t *gA, uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sums[32U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            uint64_t v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++)
        gC[(mrow * 32U + row) * n + mcol * 32U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_g_matmul_f32_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    float *gA, float *gB, uint32_t k, uint32_t n, float *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT(1024U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            float v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(mrow * 16U + row) * n + mcol * 16U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_g_matmul_f64_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    double *gA, double *gB, uint32_t k, uint32_t n, double *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT(2048U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            double v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(mrow * 16U + row) * n + mcol * 16U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_g_matmul_u32_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint32_t *gA, uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sums[16U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            uint32_t v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(mrow * 16U + row) * n + mcol * 16U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_g_matmul_u64_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint64_t *gA, uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sums[16U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            uint64_t v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++)
        gC[(mrow * 16U + row) * n + mcol * 16U + threadIdx.x] = sums[row];
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f32_tile32_rrr
*/
static void
__hoisted_g_gemm_f32_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    float *gA, float *gB, uint32_t k, uint32_t n, float *gC, float beta,
    float alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT(4096U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            float v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = mrow * 32U + row;
        uint32_t gcol_sz = mcol * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_f64_tile32_rrr
*/
static void
__hoisted_g_gemm_f64_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    double *gA, double *gB, uint32_t k, uint32_t n, double *gC, double beta,
    double alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT(8192U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            double v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = mrow * 32U + row;
        uint32_t gcol_sz = mcol * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_u32_tile32_rrr
*/
static void
__hoisted_g_gemm_u32_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint32_t *gA, uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC,
    uint32_t beta, uint32_t alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sums[32U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            uint32_t v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = mrow * 32U + row;
        uint32_t gcol_sz = mcol * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(32)
/**
  hoisted when extracting g_gemm_u64_tile32_rrr
*/
static void
__hoisted_g_gemm_u64_tile32_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint64_t *gA, uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC,
    uint64_t beta, uint64_t alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sums[32U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 32U; i++) {
            uint32_t ci = i;
            ar1[ci * 32U + threadIdx.x] =
                gA[(32U * mrow + ci) * k + 32U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 32U + threadIdx.x] =
                gB[(32U * __anf0 + ci1) * n + 32U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 32U; sk++) {
            uint32_t i1 = 0U;
            uint64_t v2 = ar2[sk * 32U + threadIdx.x];
            for (; i1 < 32U; i1++)
                sums[i1] += ar1[i1 * 32U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 32U; row++) {
        uint32_t grow_sz = mrow * 32U + row;
        uint32_t gcol_sz = mcol * 32U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f32_tile16_rrr
*/
static void
__hoisted_g_gemm_f32_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    float *gA, float *gB, uint32_t k, uint32_t n, float *gC, float beta,
    float alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT(1024U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            float v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = mrow * 16U + row;
        uint32_t gcol_sz = mcol * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_f64_tile16_rrr
*/
static void
__hoisted_g_gemm_f64_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    double *gA, double *gB, uint32_t k, uint32_t n, double *gC, double beta,
    double alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT(2048U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            double v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = mrow * 16U + row;
        uint32_t gcol_sz = mcol * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_u32_tile16_rrr
*/
static void
__hoisted_g_gemm_u32_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint32_t *gA, uint32_t *gB, uint32_t k, uint32_t n, uint32_t *gC,
    uint32_t beta, uint32_t alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint32_t sums[16U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            uint32_t v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = mrow * 16U + row;
        uint32_t gcol_sz = mcol * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

__global__ __launch_bounds__(16)
/**
  hoisted when extracting g_gemm_u64_tile16_rrr
*/
static void
__hoisted_g_gemm_u64_tile16_rrr_0(uint32_t nn, uint32_t mm, uint32_t kk,
    uint64_t *gA, uint64_t *gB, uint32_t k, uint32_t n, uint64_t *gC,
    uint64_t beta, uint64_t alpha)
{
    KRML_MAYBE_UNUSED_VAR(mm);
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint32_t mrow = blockIdx.x / nn;
    uint32_t mcol = blockIdx.x % nn;
    uint64_t sums[16U] = {0U};
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        __syncthreads();
        uint32_t __anf0 = bk;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t ci = i;
            ar1[ci * 16U + threadIdx.x] =
                gA[(16U * mrow + ci) * k + 16U * __anf0 + threadIdx.x];
            uint32_t ci1 = i;
            ar2[ci1 * 16U + threadIdx.x] =
                gB[(16U * __anf0 + ci1) * n + 16U * mcol + threadIdx.x];
        }
        __syncthreads();
        uint32_t sk = 0U;
        for (; sk < 16U; sk++) {
            uint32_t i1 = 0U;
            uint64_t v2 = ar2[sk * 16U + threadIdx.x];
            for (; i1 < 16U; i1++)
                sums[i1] += ar1[i1 * 16U + sk] * v2;
        }
    }
    uint32_t row = 0U;
    for (; row < 16U; row++) {
        uint32_t grow_sz = mrow * 16U + row;
        uint32_t gcol_sz = mcol * 16U + threadIdx.x;
        gC[grow_sz * n + gcol_sz] =
            beta * gC[grow_sz * n + gcol_sz] + alpha * sums[row];
    }
}

void Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_matmul_f32_tile32_rrr_0, mm * nn, 32U, 8192U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_matmul_f64_tile32_rrr_0, mm * nn, 32U, 16384U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(uint32_t m, uint32_t n,
    uint32_t k, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_matmul_u32_tile32_rrr_0, mm * nn, 32U, 8192U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(uint32_t m, uint32_t n,
    uint32_t k, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_matmul_u64_tile32_rrr_0, mm * nn, 32U, 16384U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_matmul_f32_tile16_rrr_0, mm * nn, 16U, 2048U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_matmul_f64_tile16_rrr_0, mm * nn, 16U, 4096U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(uint32_t m, uint32_t n,
    uint32_t k, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_matmul_u32_tile16_rrr_0, mm * nn, 16U, 2048U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(uint32_t m, uint32_t n,
    uint32_t k, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_matmul_u64_tile16_rrr_0, mm * nn, 16U, 4096U, s, nn,
        mm, kk, gA, gB, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_f32_tile32_rrr_0, mm * nn, 32U, 8192U, s, nn, mm,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_f64_tile32_rrr_0, mm * nn, 32U, 16384U, s, nn,
        mm, kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(uint32_t alpha,
    uint32_t beta, uint32_t m, uint32_t n, uint32_t k, uint32_t *gA,
    uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_g_gemm_u32_tile32_rrr_0, mm * nn, 32U, 8192U, s, nn, mm,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(uint64_t alpha,
    uint64_t beta, uint32_t m, uint32_t n, uint32_t k, uint64_t *gA,
    uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(16384U);
    KPR_KCALL(__hoisted_g_gemm_u64_tile32_rrr_0, mm * nn, 32U, 16384U, s, nn,
        mm, kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_gemm_f32_tile16_rrr_0, mm * nn, 16U, 2048U, s, nn, mm,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_f64_tile16_rrr_0, mm * nn, 16U, 4096U, s, nn, mm,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(uint32_t alpha,
    uint32_t beta, uint32_t m, uint32_t n, uint32_t k, uint32_t *gA,
    uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(2048U);
    KPR_KCALL(__hoisted_g_gemm_u32_tile16_rrr_0, mm * nn, 16U, 2048U, s, nn, mm,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(uint64_t alpha,
    uint64_t beta, uint32_t m, uint32_t n, uint32_t k, uint64_t *gA,
    uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_g_gemm_u64_tile16_rrr_0, mm * nn, 16U, 4096U, s, nn, mm,
        kk, gA, gB, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
