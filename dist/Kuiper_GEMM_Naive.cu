
#include "Kuiper_GEMM_Naive.h"

__global__
/**
  hoisted when extracting matmul_f32_rrr
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, float *gA, float *gB,
                        float *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

float
*Kuiper_GEMM_Naive_matmul_f32_rrr(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_f64_rrr
*/
static void __hoisted_1(uint32_t shared, uint32_t cols, double *gA, double *gB,
                        double *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    double sum = 0.0l;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

double
*Kuiper_GEMM_Naive_matmul_f64_rrr(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_1, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_u32_rrr
*/
static void
__hoisted_2(uint32_t shared, uint32_t cols, uint32_t *gA, uint32_t *gB,
            uint32_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint32_t sum = 0U;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

uint32_t
    * Kuiper_GEMM_Naive_matmul_u32_rrr(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_2, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_u64_rrr
*/
static void
__hoisted_3(uint32_t shared, uint32_t cols, uint64_t *gA, uint64_t *gB,
            uint64_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint64_t sum = 0ULL;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

uint64_t
    * Kuiper_GEMM_Naive_matmul_u64_rrr(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_3, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_f32_ccc
*/
static void
__hoisted_4(uint32_t rows, uint32_t shared, uint32_t cols, float *gA, float *gB,
            float *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

float
*Kuiper_GEMM_Naive_matmul_f32_ccc(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_4, rows * cols, 1U, 0U, rows, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_f64_ccc
*/
static void
__hoisted_5(uint32_t rows, uint32_t shared, uint32_t cols, double *gA,
            double *gB, double *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    double sum = 0.0l;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

double
*Kuiper_GEMM_Naive_matmul_f64_ccc(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_5, rows * cols, 1U, 0U, rows, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_u32_ccc
*/
static void
__hoisted_6(uint32_t rows,
            uint32_t shared,
            uint32_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint32_t sum = 0U;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

uint32_t
    * Kuiper_GEMM_Naive_matmul_u32_ccc(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_6, rows * cols, 1U, 0U, rows, shared, cols, gA, gB, gC);
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
  hoisted when extracting matmul_u64_ccc
*/
static void
__hoisted_7(uint32_t rows,
            uint32_t shared,
            uint32_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint64_t sum = 0ULL;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

uint64_t
    * Kuiper_GEMM_Naive_matmul_u64_ccc(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_7, rows * cols, 1U, 0U, rows, shared, cols, gA, gB, gC);
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
  hoisted when extracting g_matmul_f32_rrr
*/
static void __hoisted_8(uint32_t shared, uint32_t cols, float *gA, float *gB,
                        float *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_f32_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   float *gA, float *gB, float *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_8, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void __hoisted_9(uint32_t shared, uint32_t cols, double *gA, double *gB,
                        double *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    double sum = 0.0l;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_f64_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   double *gA, double *gB, double *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_9, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_10(uint32_t shared, uint32_t cols, uint32_t *gA, uint32_t *gB,
             uint32_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint32_t sum = 0U;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_u32_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_10, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_11(uint32_t shared, uint32_t cols, uint64_t *gA, uint64_t *gB,
             uint64_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint64_t sum = 0ULL;
    for (; k < shared; k++)
        sum += gA[trow * shared + k] * gB[k * cols + tcol];
    gC[trow * cols + tcol] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_u64_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_11, rows * cols, 1U, 0U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void
__hoisted_12(uint32_t rows, uint32_t shared, uint32_t cols, float *gA,
             float *gB, float *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_f32_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   float *gA, float *gB, float *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_12, rows * cols, 1U, 0U, rows, shared, cols, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void
__hoisted_13(uint32_t rows, uint32_t shared, uint32_t cols, double *gA,
             double *gB, double *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    double sum = 0.0l;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_f64_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   double *gA, double *gB, double *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_13, rows * cols, 1U, 0U, rows, shared, cols, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_ccc
*/
static void
__hoisted_14(uint32_t rows,
             uint32_t shared,
             uint32_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint32_t sum = 0U;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_u32_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_14, rows * cols, 1U, 0U, rows, shared, cols, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_ccc
*/
static void
__hoisted_15(uint32_t rows,
             uint32_t shared,
             uint32_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t trow = blockIdx.x / cols;
    uint32_t tcol = blockIdx.x % cols;
    uint32_t k = 0U;
    uint64_t sum = 0ULL;
    for (; k < shared; k++)
        sum += gA[k * rows + trow] * gB[tcol * shared + k];
    gC[tcol * rows + trow] = sum;
}

void
Kuiper_GEMM_Naive_g_matmul_u64_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_15, rows * cols, 1U, 0U, rows, shared, cols, gA, gB,
              gC);
    MUST(cudaDeviceSynchronize());
}
