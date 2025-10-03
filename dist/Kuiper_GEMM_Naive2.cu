
#include "Kuiper_GEMM_Naive2.h"

__global__
/**
  hoisted when extracting matmul_f32_rrr
*/
static void
__hoisted_0(uint32_t rows,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        float_t sum = 0.0f;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

float_t
    * Kuiper_GEMM_Naive2_matmul_f32_rrr(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, float_t *a, float_t *b)
{
    float_t *gA = (float_t *) KPR_GPU_ALLOC(4U, rows * shared);
    float_t *gB = (float_t *) KPR_GPU_ALLOC(4U, shared * cols);
    float_t *gC = (float_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_0,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
    KRML_CHECK_SIZE(sizeof(float_t), rows * cols);
    float_t *c = (float_t *) KRML_HOST_MALLOC(sizeof(float_t) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(float_t));
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
static void
__hoisted_1(uint32_t rows,
            uint32_t shared,
            uint32_t cols, double_t *gA, double_t *gB, double_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        double_t sum = 0.0l;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

double_t
    * Kuiper_GEMM_Naive2_matmul_f64_rrr(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, double_t *a, double_t *b)
{
    double_t *gA = (double_t *) KPR_GPU_ALLOC(8U, rows * shared);
    double_t *gB = (double_t *) KPR_GPU_ALLOC(8U, shared * cols);
    double_t *gC = (double_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_1,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
    KRML_CHECK_SIZE(sizeof(double_t), rows * cols);
    double_t *c =
        (double_t *) KRML_HOST_MALLOC(sizeof(double_t) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(double_t));
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
__hoisted_2(uint32_t rows,
            uint32_t shared,
            uint32_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint32_t sum = 0U;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

uint32_t
    * Kuiper_GEMM_Naive2_matmul_u32_rrr(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_2,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
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
__hoisted_3(uint32_t rows,
            uint32_t shared,
            uint32_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint64_t sum = 0ULL;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

uint64_t
    * Kuiper_GEMM_Naive2_matmul_u64_rrr(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_3,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
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
__hoisted_4(uint32_t rows,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        float_t sum = 0.0f;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

float_t
    * Kuiper_GEMM_Naive2_matmul_f32_ccc(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, float_t *a, float_t *b)
{
    float_t *gA = (float_t *) KPR_GPU_ALLOC(4U, rows * shared);
    float_t *gB = (float_t *) KPR_GPU_ALLOC(4U, shared * cols);
    float_t *gC = (float_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_4,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
    KRML_CHECK_SIZE(sizeof(float_t), rows * cols);
    float_t *c = (float_t *) KRML_HOST_MALLOC(sizeof(float_t) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(float_t));
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
__hoisted_5(uint32_t rows,
            uint32_t shared,
            uint32_t cols, double_t *gA, double_t *gB, double_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        double_t sum = 0.0l;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

double_t
    * Kuiper_GEMM_Naive2_matmul_f64_ccc(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, double_t *a, double_t *b)
{
    double_t *gA = (double_t *) KPR_GPU_ALLOC(8U, rows * shared);
    double_t *gB = (double_t *) KPR_GPU_ALLOC(8U, shared * cols);
    double_t *gC = (double_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_5,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
    KRML_CHECK_SIZE(sizeof(double_t), rows * cols);
    double_t *c =
        (double_t *) KRML_HOST_MALLOC(sizeof(double_t) * (rows * cols));
    if (c != NULL)
        memset(c, 0U, rows * cols * sizeof(double_t));
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
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint32_t sum = 0U;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

uint32_t
    * Kuiper_GEMM_Naive2_matmul_u32_ccc(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_6,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
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
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint64_t sum = 0ULL;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

uint64_t
    * Kuiper_GEMM_Naive2_matmul_u64_ccc(uint32_t rows,
                                        uint32_t shared,
                                        uint32_t cols, uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_KCALL(__hoisted_7,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
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
static void
__hoisted_8(uint32_t rows,
            uint32_t shared,
            uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        float_t sum = 0.0f;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_f32_rrr(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float_t *gA, float_t *gB, float_t *gC)
{
    KPR_KCALL(__hoisted_8,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_9(uint32_t rows,
            uint32_t shared,
            uint32_t cols, double_t *gA, double_t *gB, double_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        double_t sum = 0.0l;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_f64_rrr(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    double_t *gA, double_t *gB, double_t *gC)
{
    KPR_KCALL(__hoisted_9,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_10(uint32_t rows,
             uint32_t shared,
             uint32_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint32_t sum = 0U;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_u32_rrr(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_KCALL(__hoisted_10,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_11(uint32_t rows,
             uint32_t shared,
             uint32_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint64_t sum = 0ULL;
        for (; k < shared; k += 1U)
            sum +=
                gA[(1024U * blockIdx.x + threadIdx.x) / cols * shared + k] *
                gB[k * cols + (1024U * blockIdx.x + threadIdx.x) % cols];
        gC[1024U * blockIdx.x + threadIdx.x] = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_u64_rrr(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_KCALL(__hoisted_11,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void
__hoisted_12(uint32_t rows,
             uint32_t shared,
             uint32_t cols, float_t *gA, float_t *gB, float_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        float_t sum = 0.0f;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_f32_ccc(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    float_t *gA, float_t *gB, float_t *gC)
{
    KPR_KCALL(__hoisted_12,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void
__hoisted_13(uint32_t rows,
             uint32_t shared,
             uint32_t cols, double_t *gA, double_t *gB, double_t *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        double_t sum = 0.0l;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_f64_ccc(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    double_t *gA, double_t *gB, double_t *gC)
{
    KPR_KCALL(__hoisted_13,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
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
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint32_t sum = 0U;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_u32_ccc(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_KCALL(__hoisted_14,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
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
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t k = 0U;
        uint64_t sum = 0ULL;
        for (; k < shared; k += 1U)
            sum +=
                gA[k * rows + (1024U * blockIdx.x + threadIdx.x) / cols] *
                gB[(1024U * blockIdx.x + threadIdx.x) % cols * shared + k];
        gC[(1024U * blockIdx.x + threadIdx.x) % cols * rows +
           (1024U * blockIdx.x + threadIdx.x) / cols]
            = sum;
    }
}

void
Kuiper_GEMM_Naive2_g_matmul_u64_ccc(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_KCALL(__hoisted_15,
              (rows * cols + 1023U) / 1024U,
              1024U, 0U, rows, shared, cols, gA, gB, gC);
    cudaDeviceSynchronize();
}
