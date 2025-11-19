
#include "Kuiper_GEMM_BlockTiling1D.h"

__global__
/**
  hoisted when extracting matmul_f32_tile32_rrr
*/
static void
__hoisted_0(uint32_t shared,
            uint32_t cols,
            float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile32_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_0, mrows * mcols, 32U, 8192U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(float), rows * cols);
    float *c = (float *)KRML_HOST_MALLOC(sizeof(float) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(float));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile32_rrr
*/
static void
__hoisted_1(uint32_t shared,
            uint32_t cols,
            double *gA,
            double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile32_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_1, mrows * mcols, 32U, 16384U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(double), rows * cols);
    double *c = (double *)KRML_HOST_MALLOC(sizeof(double) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(double));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile32_rrr
*/
static void
__hoisted_2(uint32_t shared,
            uint32_t cols,
            uint32_t *gA,
            uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile32_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_2, mrows * mcols, 32U, 8192U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint32_t), rows * cols);
    uint32_t *c = (uint32_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint32_t));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile32_rrr
*/
static void
__hoisted_3(uint32_t shared,
            uint32_t cols,
            uint64_t *gA,
            uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile32_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_3, mrows * mcols, 32U, 16384U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint64_t), rows * cols);
    uint64_t *c = (uint64_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint64_t));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile32_ccc
*/
static void
__hoisted_4(uint32_t rows,
            uint32_t shared,
            float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile32_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_4, mrows * mcols, 32U, 8192U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(float), rows * cols);
    float *c = (float *)KRML_HOST_MALLOC(sizeof(float) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(float));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile32_ccc
*/
static void
__hoisted_5(uint32_t rows,
            uint32_t shared,
            double *gA,
            double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile32_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_5, mrows * mcols, 32U, 16384U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(double), rows * cols);
    double *c = (double *)KRML_HOST_MALLOC(sizeof(double) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(double));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile32_ccc
*/
static void
__hoisted_6(uint32_t rows,
            uint32_t shared,
            uint32_t *gA,
            uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile32_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_6, mrows * mcols, 32U, 8192U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint32_t), rows * cols);
    uint32_t *c = (uint32_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint32_t));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile32_ccc
*/
static void
__hoisted_7(uint32_t rows,
            uint32_t shared,
            uint64_t *gA,
            uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile32_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_7, mrows * mcols, 32U, 16384U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint64_t), rows * cols);
    uint64_t *c = (uint64_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint64_t));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile16_rrr
*/
static void
__hoisted_8(uint32_t shared,
            uint32_t cols,
            float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile16_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_8, mrows * mcols, 16U, 2048U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(float), rows * cols);
    float *c = (float *)KRML_HOST_MALLOC(sizeof(float) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(float));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile16_rrr
*/
static void
__hoisted_9(uint32_t shared,
            uint32_t cols,
            double *gA,
            double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile16_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_9, mrows * mcols, 16U, 4096U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(double), rows * cols);
    double *c = (double *)KRML_HOST_MALLOC(sizeof(double) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(double));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile16_rrr
*/
static void
__hoisted_10(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile16_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_10, mrows * mcols, 16U, 2048U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint32_t), rows * cols);
    uint32_t *c = (uint32_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint32_t));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile16_rrr
*/
static void
__hoisted_11(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile16_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_11, mrows * mcols, 16U, 4096U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint64_t), rows * cols);
    uint64_t *c = (uint64_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint64_t));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile16_ccc
*/
static void
__hoisted_12(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile16_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_12, mrows * mcols, 16U, 2048U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(float), rows * cols);
    float *c = (float *)KRML_HOST_MALLOC(sizeof(float) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(float));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile16_ccc
*/
static void
__hoisted_13(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile16_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_13, mrows * mcols, 16U, 4096U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(double), rows * cols);
    double *c = (double *)KRML_HOST_MALLOC(sizeof(double) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(double));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile16_ccc
*/
static void
__hoisted_14(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile16_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_14, mrows * mcols, 16U, 2048U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint32_t), rows * cols);
    uint32_t *c = (uint32_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint32_t));
    MUST(cudaMemcpy(c, gC, 4U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile16_ccc
*/
static void
__hoisted_15(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile16_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_15, mrows * mcols, 16U, 4096U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
    KRML_CHECK_SIZE(sizeof(uint64_t), rows * cols);
    uint64_t *c = (uint64_t *) KRML_HOST_CALLOC(rows * cols, sizeof(uint64_t));
    MUST(cudaMemcpy(c, gC, 8U * (rows * cols), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    return c;
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_16(uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_16, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_16, mrows * mcols, 32U, 8192U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_17(uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_17, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_17, mrows * mcols, 32U, 16384U, shared, cols, gA, gB,
              gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_18(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t *gA,
                                                  uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_18, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_18, mrows * mcols, 32U, 8192U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_19(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t *gA,
                                                  uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_19, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_19, mrows * mcols, 32U, 16384U, shared, cols, gA, gB,
              gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_ccc
*/
static void
__hoisted_20(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_20, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_20, mrows * mcols, 32U, 8192U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_ccc
*/
static void
__hoisted_21(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_21, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_21, mrows * mcols, 32U, 16384U, rows, shared, gA, gB,
              gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_ccc
*/
static void
__hoisted_22(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t *gA,
                                                  uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_22, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_22, mrows * mcols, 32U, 8192U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_ccc
*/
static void
__hoisted_23(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t *gA,
                                                  uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_23, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_23, mrows * mcols, 32U, 16384U, rows, shared, gA, gB,
              gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_24(uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_24, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_24, mrows * mcols, 16U, 2048U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_25(uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_25, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_25, mrows * mcols, 16U, 4096U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_26(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t *gA,
                                                  uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_26, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_26, mrows * mcols, 16U, 2048U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_27(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t *gA,
                                                  uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_27, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_27, mrows * mcols, 16U, 4096U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_ccc
*/
static void
__hoisted_28(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_28, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_28, mrows * mcols, 16U, 2048U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_ccc
*/
static void
__hoisted_29(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_29, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_29, mrows * mcols, 16U, 4096U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_ccc
*/
static void
__hoisted_30(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t *gA,
                                                  uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_30, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_30, mrows * mcols, 16U, 2048U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_ccc
*/
static void
__hoisted_31(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] = sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t *gA,
                                                  uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_31, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_31, mrows * mcols, 16U, 4096U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile32_rrr
*/
static void
__hoisted_32(float alpha,
             float beta,
             uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 32U + row) * cols +
                         blockIdx.x % mcols * 32U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_32, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_32, mrows * mcols, 32U, 8192U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile32_rrr
*/
static void
__hoisted_33(double alpha,
             double beta,
             uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 32U + row) * cols +
                         blockIdx.x % mcols * 32U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_33, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_33, mrows * mcols, 32U, 16384U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile32_rrr
*/
static void
__hoisted_34(uint32_t alpha,
             uint32_t beta,
             uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 32U + row) * cols +
                         blockIdx.x % mcols * 32U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_34, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_34, mrows * mcols, 32U, 8192U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile32_rrr
*/
static void
__hoisted_35(uint64_t alpha,
             uint64_t beta,
             uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 32U + vi) * shared + vbk1 * 32U +
                   threadIdx.x];
            sa2[vi * 32U + threadIdx.x] =
                gB[(vbk1 * 32U + vi) * cols + blockIdx.x % mcols * 32U +
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
        tileC[(blockIdx.x / mcols * 32U + row) * cols +
              blockIdx.x % mcols * 32U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 32U + row) * cols +
                         blockIdx.x % mcols * 32U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_35, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_35, mrows * mcols, 32U, 16384U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile32_ccc
*/
static void
__hoisted_36(float alpha,
             float beta,
             uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4096U);
    float sums[32U];
    memset(sums, 0U, 32U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] =
            beta * tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 32U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_ccc(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_36, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_36, mrows * mcols, 32U, 8192U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile32_ccc
*/
static void
__hoisted_37(double alpha,
             double beta,
             uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8192U);
    double sums[32U];
    memset(sums, 0U, 32U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] =
            beta * tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 32U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_ccc(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_37, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_37, mrows * mcols, 32U, 16384U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile32_ccc
*/
static void
__hoisted_38(uint32_t alpha,
             uint32_t beta,
             uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4096U);
    uint32_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] =
            beta * tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 32U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_ccc(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute
         (__hoisted_38, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192U));
    KPR_KCALL(__hoisted_38, mrows * mcols, 32U, 8192U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile32_ccc
*/
static void
__hoisted_39(uint64_t alpha,
             uint64_t beta,
             uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8192U);
    uint64_t sums[32U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 32U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 32U + threadIdx.x] =
                gA[(vbk1 * 32U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 32U + vi];
            sa2[vi * 32U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 32U + threadIdx.x) * shared +
                   vbk1 * 32U + vi];
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
        tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
              blockIdx.x / mcols * 32U + row] =
            beta * tileC[(blockIdx.x % mcols * 32U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 32U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_ccc(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_39, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_39, mrows * mcols, 32U, 16384U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile16_rrr
*/
static void
__hoisted_40(float alpha,
             float beta,
             uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 16U + row) * cols +
                         blockIdx.x % mcols * 16U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_40, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_40, mrows * mcols, 16U, 2048U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile16_rrr
*/
static void
__hoisted_41(double alpha,
             double beta,
             uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 16U + row) * cols +
                         blockIdx.x % mcols * 16U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_41, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_41, mrows * mcols, 16U, 4096U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile16_rrr
*/
static void
__hoisted_42(uint32_t alpha,
             uint32_t beta,
             uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 16U + row) * cols +
                         blockIdx.x % mcols * 16U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_42, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_42, mrows * mcols, 16U, 2048U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile16_rrr
*/
static void
__hoisted_43(uint64_t alpha,
             uint64_t beta,
             uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(blockIdx.x / mcols * 16U + vi) * shared + vbk1 * 16U +
                   threadIdx.x];
            sa2[vi * 16U + threadIdx.x] =
                gB[(vbk1 * 16U + vi) * cols + blockIdx.x % mcols * 16U +
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
        tileC[(blockIdx.x / mcols * 16U + row) * cols +
              blockIdx.x % mcols * 16U + threadIdx.x] =
            beta * tileC[(blockIdx.x / mcols * 16U + row) * cols +
                         blockIdx.x % mcols * 16U + threadIdx.x]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_43, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_43, mrows * mcols, 16U, 4096U, alpha, beta, shared,
              cols, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tile16_ccc
*/
static void
__hoisted_44(float alpha,
             float beta,
             uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(1024U);
    float sums[16U];
    memset(sums, 0U, 16U * sizeof(float));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] =
            beta * tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 16U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_ccc(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA, float *gB, float *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_44, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_44, mrows * mcols, 16U, 2048U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_tile16_ccc
*/
static void
__hoisted_45(double alpha,
             double beta,
             uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(2048U);
    double sums[16U];
    memset(sums, 0U, 16U * sizeof(double));
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] =
            beta * tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 16U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_ccc(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_45, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_45, mrows * mcols, 16U, 4096U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_tile16_ccc
*/
static void
__hoisted_46(uint32_t alpha,
             uint32_t beta,
             uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(1024U);
    uint32_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] =
            beta * tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 16U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_ccc(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t *gA,
                                                uint32_t *gB, uint32_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute
         (__hoisted_46, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048U));
    KPR_KCALL(__hoisted_46, mrows * mcols, 16U, 2048U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_tile16_ccc
*/
static void
__hoisted_47(uint64_t alpha,
             uint64_t beta,
             uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(2048U);
    uint64_t sums[16U] = { 0U };
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        __syncthreads();
        uint32_t vbk1 = bk;
        uint32_t i0 = 0U;
        for (; i0 < 16U; i0++) {
            uint32_t vi = i0;
            sa1[vi * 16U + threadIdx.x] =
                gA[(vbk1 * 16U + threadIdx.x) * rows +
                   blockIdx.x / mcols * 16U + vi];
            sa2[vi * 16U + threadIdx.x] =
                gB[(blockIdx.x % mcols * 16U + threadIdx.x) * shared +
                   vbk1 * 16U + vi];
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
        tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
              blockIdx.x / mcols * 16U + row] =
            beta * tileC[(blockIdx.x % mcols * 16U + threadIdx.x) * rows +
                         blockIdx.x / mcols * 16U + row]
            + alpha * sums[row];
}

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_ccc(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t *gA,
                                                uint64_t *gB, uint64_t *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_47, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_47, mrows * mcols, 16U, 4096U, alpha, beta, rows,
              shared, gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}
