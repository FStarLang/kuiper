
#include "Klas_RowSoftmax.h"

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void __hoisted_row_softmax_rm_f32_0(uint32_t n, float *a, float *maxs,
                                           uint32_t nthm)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float acc = a[blockIdx.x * n + threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < n; idx += nthm)
        acc = fmaxf(acc, a[blockIdx.x * n + idx]);
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < nthm; n1++) {
        uint32_t __anf02 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa1[threadIdx.x] = fmaxf(sa1[threadIdx.x], sa1[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void __hoisted_row_softmax_rm_f32_1(uint32_t m, uint32_t n, float *a,
                                           float *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        a[row * n + col] = maxs[row] - a[row * n + col];
    }
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void __hoisted_row_softmax_rm_f32_2(uint32_t n, float *a, float *sums)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U) {
        float v_ = expf(a[blockIdx.x * n + idx]);
        acc += v_;
    }
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void __hoisted_row_softmax_rm_f32_3(uint32_t m, uint32_t n, float *a,
                                           float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        float vb = a[row * n + col];
        uint32_t ni = row * n + col;
        a[ni] = expf(sums[row]) / vb;
    }
}

void Klas_RowSoftmax_row_softmax_rm_f32(uint32_t m, uint32_t n, float *a)
{
    float *maxs = (float *)KPR_GPU_ALLOC(sizeof(float), m);
    float *sums = (float *)KPR_GPU_ALLOC(sizeof(float), m);
    uint32_t nthm = 1024U <= n ? 1024U : n;
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_row_softmax_rm_f32_0, m, nthm, 4U * nthm, n, a, maxs,
              nthm);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_row_softmax_rm_f32_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, a, maxs);
    MUST(cudaDeviceSynchronize());
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f32_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_row_softmax_rm_f32_2, m, 1024U, 4096U, n, a, sums);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_row_softmax_rm_f32_3,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, a, sums);
    MUST(cudaDeviceSynchronize());
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

__global__
/**
  hoisted when extracting row_softmax_rm_f64
*/
static void __hoisted_row_softmax_rm_f64_0(uint32_t n, double *a, double *maxs,
                                           uint32_t nthm)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double acc = a[blockIdx.x * n + threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < n; idx += nthm)
        acc = fmax(acc, a[blockIdx.x * n + idx]);
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < nthm; n1++) {
        uint32_t __anf02 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa1[threadIdx.x] = fmax(sa1[threadIdx.x], sa1[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_softmax_rm_f64
*/
static void __hoisted_row_softmax_rm_f64_1(uint32_t m, uint32_t n, double *a,
                                           double *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        a[row * n + col] = maxs[row] - a[row * n + col];
    }
}

__global__
/**
  hoisted when extracting row_softmax_rm_f64
*/
static void __hoisted_row_softmax_rm_f64_2(uint32_t n, double *a, double *sums)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U) {
        double v_ = exp(a[blockIdx.x * n + idx]);
        acc += v_;
    }
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_softmax_rm_f64
*/
static void __hoisted_row_softmax_rm_f64_3(uint32_t m, uint32_t n, double *a,
                                           double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        double vb = a[row * n + col];
        uint32_t ni = row * n + col;
        a[ni] = exp(sums[row]) / vb;
    }
}

void Klas_RowSoftmax_row_softmax_rm_f64(uint32_t m, uint32_t n, double *a)
{
    double *maxs = (double *)KPR_GPU_ALLOC(sizeof(double), m);
    double *sums = (double *)KPR_GPU_ALLOC(sizeof(double), m);
    uint32_t nthm = 1024U <= n ? 1024U : n;
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_row_softmax_rm_f64_0, m, nthm, 8U * nthm, n, a, maxs,
              nthm);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_row_softmax_rm_f64_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, a, maxs);
    MUST(cudaDeviceSynchronize());
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f64_2,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_row_softmax_rm_f64_2, m, 1024U, 8192U, n, a, sums);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_row_softmax_rm_f64_3,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, a, sums);
    MUST(cudaDeviceSynchronize());
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}
