
#include "Klas_RowSoftmax.h"

typedef struct __uint32_t__uint32_t_______s {
    uint32_t fst;
    uint32_t snd;
} __uint32_t__uint32_t______;

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void __hoisted_row_softmax_rm_f32_0(uint32_t n, float *a, float *sums)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U) {
        __uint32_t__uint32_t______ scrut = {.fst = blockIdx.x,.snd = idx };
        float v_ = expf(a[scrut.fst * n + scrut.snd]);
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
static void __hoisted_row_softmax_rm_f32_1(uint32_t m, uint32_t n, float *a,
                                           float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        float v = sums[row];
        uint32_t ni = row * n + col;
        a[ni] = expf(a[row * n + col]) / v;
    }
}

void Klas_RowSoftmax_row_softmax_rm_f32(uint32_t m, uint32_t n, float *a)
{
    float *sums = (float *)KPR_GPU_ALLOC(sizeof(float), m);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_row_softmax_rm_f32_0, m, 1024U, 4096U, n, a, sums);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_row_softmax_rm_f32_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, a, sums);
    MUST(cudaDeviceSynchronize());
    MUST(cudaFree(sums));
}

__global__
/**
  hoisted when extracting row_softmax_rm_f64
*/
static void __hoisted_row_softmax_rm_f64_0(uint32_t n, double *a, double *sums)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U) {
        __uint32_t__uint32_t______ scrut = {.fst = blockIdx.x,.snd = idx };
        double v_ = exp(a[scrut.fst * n + scrut.snd]);
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
static void __hoisted_row_softmax_rm_f64_1(uint32_t m, uint32_t n, double *a,
                                           double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        double v = sums[row];
        uint32_t ni = row * n + col;
        a[ni] = exp(a[row * n + col]) / v;
    }
}

void Klas_RowSoftmax_row_softmax_rm_f64(uint32_t m, uint32_t n, double *a)
{
    double *sums = (double *)KPR_GPU_ALLOC(sizeof(double), m);
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_row_softmax_rm_f64_0, m, 1024U, 8192U, n, a, sums);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_row_softmax_rm_f64_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, m, n, a, sums);
    MUST(cudaDeviceSynchronize());
    MUST(cudaFree(sums));
}
