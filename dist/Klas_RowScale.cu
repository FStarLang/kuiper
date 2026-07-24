
#include "Klas_RowScale.h"

__global__
/**
  hoisted when extracting rowscale_f16_rowmajor
*/
static void __hoisted_rowscale_f16_rowmajor_0(uint32_t m, uint32_t n, half *a,
                                              half *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t ni = row * n + col;
        b[ni] = __hmul(a[row], b[row * n + col]);
    }
}

void Klas_RowScale_rowscale_f16_rowmajor(uint32_t m, uint32_t n, half *a,
                                         half *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f16_rowmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting rowscale_f16_colmajor
*/
static void __hoisted_rowscale_f16_colmajor_0(uint32_t m, uint32_t n, half *a,
                                              half *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t ni = col * m + row;
        b[ni] = __hmul(a[row], b[col * m + row]);
    }
}

void Klas_RowScale_rowscale_f16_colmajor(uint32_t m, uint32_t n, half *a,
                                         half *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f16_colmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting rowscale_f32_rowmajor
*/
static void __hoisted_rowscale_f32_rowmajor_0(uint32_t m, uint32_t n, float *a,
                                              float *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        b[row * n + col] *= a[row];
    }
}

void Klas_RowScale_rowscale_f32_rowmajor(uint32_t m, uint32_t n, float *a,
                                         float *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f32_rowmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting rowscale_f32_colmajor
*/
static void __hoisted_rowscale_f32_colmajor_0(uint32_t m, uint32_t n, float *a,
                                              float *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        b[col * m + row] *= a[row];
    }
}

void Klas_RowScale_rowscale_f32_colmajor(uint32_t m, uint32_t n, float *a,
                                         float *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f32_colmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting rowscale_f64_rowmajor
*/
static void __hoisted_rowscale_f64_rowmajor_0(uint32_t m, uint32_t n, double *a,
                                              double *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        b[row * n + col] *= a[row];
    }
}

void Klas_RowScale_rowscale_f64_rowmajor(uint32_t m, uint32_t n, double *a,
                                         double *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f64_rowmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting rowscale_f64_colmajor
*/
static void __hoisted_rowscale_f64_colmajor_0(uint32_t m, uint32_t n, double *a,
                                              double *b)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        b[col * m + row] *= a[row];
    }
}

void Klas_RowScale_rowscale_f64_colmajor(uint32_t m, uint32_t n, double *a,
                                         double *b)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rowscale_f64_colmajor_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, a, b);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
