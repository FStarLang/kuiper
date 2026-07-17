
#include "Klas_GEMM_Naive2.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_g_matmul_f32_rrr_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, float *gA, float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float sum = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[trow * k + vk] * gB[vk * n + tcol];
        }
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f32_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_g_matmul_f64_rrr_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, double *gA, double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double sum = 0.0;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[trow * k + vk] * gB[vk * n + tcol];
        }
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f64_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_g_matmul_u32_rrr_0(uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint32_t sum = 0U;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[trow * k + vk] * gB[vk * n + tcol];
        }
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u32_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_KCALL(__hoisted_g_matmul_u32_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_g_matmul_u64_rrr_0(uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint64_t sum = 0ULL;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[trow * k + vk] * gB[vk * n + tcol];
        }
        gC[trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u64_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_KCALL(__hoisted_g_matmul_u64_rrr_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void
__hoisted_g_matmul_f32_ccc_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, float *gA, float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float sum = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[vk * m + trow] * gB[tcol * k + vk];
        }
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f32_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f32_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void
__hoisted_g_matmul_f64_ccc_0(uint32_t m,
                             uint32_t n,
                             uint32_t k, double *gA, double *gB, double *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        double sum = 0.0;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[vk * m + trow] * gB[tcol * k + vk];
        }
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_f64_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC)
{
    KPR_KCALL(__hoisted_g_matmul_f64_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_ccc
*/
static void
__hoisted_g_matmul_u32_ccc_0(uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint32_t sum = 0U;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[vk * m + trow] * gB[tcol * k + vk];
        }
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u32_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_KCALL(__hoisted_g_matmul_u32_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_ccc
*/
static void
__hoisted_g_matmul_u64_ccc_0(uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        uint64_t sum = 0ULL;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum += gA[vk * m + trow] * gB[tcol * k + vk];
        }
        gC[tcol * m + trow] = sum;
    }
}

void
Klas_GEMM_Naive2_g_matmul_u64_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_KCALL(__hoisted_g_matmul_u64_ccc_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, k, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting batched_matmul_f32
*/
static void
__hoisted_batched_matmul_f32_0(uint32_t batch,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, float *a, float *b, float *c)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * m * n) {
        uint32_t page = (1024U * blockIdx.x + threadIdx.x) % batch;
        uint32_t rest = (1024U * blockIdx.x + threadIdx.x) / batch;
        uint32_t trow = rest / n;
        uint32_t tcol = rest % n;
        uint32_t k1 = 0U;
        float sum = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum +=
                a[page * m * k + trow * k + vk] * b[page * k * n + vk * n +
                                                    tcol];
        }
        c[page * m * n + trow * n + tcol] = sum;
    }
}

void
Klas_GEMM_Naive2_batched_matmul_f32(uint32_t batch,
                                    uint32_t m,
                                    uint32_t n,
                                    uint32_t k, float *a, float *b, float *c)
{
    KPR_KCALL(__hoisted_batched_matmul_f32_0,
              batch * m * n / 1024U + (uint32_t) (batch * m * n % 1024U != 0U),
              1024U, 0U, batch, m, n, k, a, b, c);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting batched_gemm_f32
*/
static void
__hoisted_batched_gemm_f32_0(float alpha,
                             float beta,
                             uint32_t batch,
                             uint32_t m,
                             uint32_t n,
                             uint32_t k, float *a, float *b, float *c)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * m * n) {
        uint32_t page = (1024U * blockIdx.x + threadIdx.x) % batch;
        uint32_t rest = (1024U * blockIdx.x + threadIdx.x) / batch;
        uint32_t trow = rest / n;
        uint32_t tcol = rest % n;
        uint32_t k1 = 0U;
        float sum = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            sum +=
                a[page * m * k + trow * k + vk] * b[page * k * n + vk * n +
                                                    tcol];
        }
        float s = sum;
        c[page * m * n + trow * n + tcol] =
            beta * c[page * m * n + trow * n + tcol] + alpha * s;
    }
}

void
Klas_GEMM_Naive2_batched_gemm_f32(float alpha,
                                  float beta,
                                  uint32_t batch,
                                  uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *a, float *b, float *c)
{
    KPR_KCALL(__hoisted_batched_gemm_f32_0,
              batch * m * n / 1024U + (uint32_t) (batch * m * n % 1024U != 0U),
              1024U, 0U, alpha, beta, batch, m, n, k, a, b, c);
    MUST(cudaDeviceSynchronize());
}
