
#include "Kuiper_GEMM_Tiled.h"

__global__
/**
  hoisted when extracting matmul_f32_rrr
*/
static void
__hoisted_0(uint32_t tile,
            uint32_t shared,
            uint32_t cols,
            float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

float
*Kuiper_GEMM_Tiled_matmul_f32_rrr(uint32_t tile,
                                  uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_0, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
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
static void
__hoisted_1(uint32_t tile,
            uint32_t shared,
            uint32_t cols,
            double *gA,
            double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

double
*Kuiper_GEMM_Tiled_matmul_f64_rrr(uint32_t tile,
                                  uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_1, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
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
__hoisted_2(uint32_t tile,
            uint32_t shared,
            uint32_t cols,
            uint32_t *gA,
            uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

uint32_t
    * Kuiper_GEMM_Tiled_matmul_u32_rrr(uint32_t tile,
                                       uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_2, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_2, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
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
__hoisted_3(uint32_t tile,
            uint32_t shared,
            uint32_t cols,
            uint64_t *gA,
            uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

uint64_t
    * Kuiper_GEMM_Tiled_matmul_u64_rrr(uint32_t tile,
                                       uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_3, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_3, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
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
__hoisted_4(uint32_t tile,
            uint32_t rows,
            uint32_t shared,
            float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

float
*Kuiper_GEMM_Tiled_matmul_f32_ccc(uint32_t tile,
                                  uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, float *a, float *b)
{
    float *gA = (float *)KPR_GPU_ALLOC(4U, rows * shared);
    float *gB = (float *)KPR_GPU_ALLOC(4U, shared * cols);
    float *gC = (float *)KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_4, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_4, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
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
__hoisted_5(uint32_t tile,
            uint32_t rows,
            uint32_t shared,
            double *gA,
            double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

double
*Kuiper_GEMM_Tiled_matmul_f64_ccc(uint32_t tile,
                                  uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, double *a, double *b)
{
    double *gA = (double *)KPR_GPU_ALLOC(8U, rows * shared);
    double *gB = (double *)KPR_GPU_ALLOC(8U, shared * cols);
    double *gC = (double *)KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_5, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_5, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
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
__hoisted_6(uint32_t tile,
            uint32_t rows,
            uint32_t shared,
            uint32_t *gA,
            uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

uint32_t
    * Kuiper_GEMM_Tiled_matmul_u32_ccc(uint32_t tile,
                                       uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint32_t *a, uint32_t *b)
{
    uint32_t *gA = (uint32_t *) KPR_GPU_ALLOC(4U, rows * shared);
    uint32_t *gB = (uint32_t *) KPR_GPU_ALLOC(4U, shared * cols);
    uint32_t *gC = (uint32_t *) KPR_GPU_ALLOC(4U, rows * cols);
    MUST(cudaMemcpy(gA, a, 4U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 4U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_6, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_6, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
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
__hoisted_7(uint32_t tile,
            uint32_t rows,
            uint32_t shared,
            uint64_t *gA,
            uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

uint64_t
    * Kuiper_GEMM_Tiled_matmul_u64_ccc(uint32_t tile,
                                       uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols, uint64_t *a, uint64_t *b)
{
    uint64_t *gA = (uint64_t *) KPR_GPU_ALLOC(8U, rows * shared);
    uint64_t *gB = (uint64_t *) KPR_GPU_ALLOC(8U, shared * cols);
    uint64_t *gC = (uint64_t *) KPR_GPU_ALLOC(8U, rows * cols);
    MUST(cudaMemcpy(gA, a, 8U * (rows * shared), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gB, b, 8U * (shared * cols), cudaMemcpyHostToDevice));
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_7, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
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
  hoisted when extracting matmul_f32_tile32_rrr
*/
static void
__hoisted_8(uint32_t shared,
            uint32_t cols,
            float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

float
*Kuiper_GEMM_Tiled_matmul_f32_tile32_rrr(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, float *a, float *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_8, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_8, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_9(uint32_t shared,
            uint32_t cols,
            double *gA,
            double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

double
*Kuiper_GEMM_Tiled_matmul_f64_tile32_rrr(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, double *a, double *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_9, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_9, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_10(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

uint32_t
    * Kuiper_GEMM_Tiled_matmul_u32_tile32_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_10, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_10, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_11(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

uint64_t
    * Kuiper_GEMM_Tiled_matmul_u64_tile32_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_11, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_11, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_12(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

float
*Kuiper_GEMM_Tiled_matmul_f32_tile32_ccc(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, float *a, float *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_12, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_12, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_13(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

double
*Kuiper_GEMM_Tiled_matmul_f64_tile32_ccc(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, double *a, double *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_13, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_13, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_14(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

uint32_t
    * Kuiper_GEMM_Tiled_matmul_u32_tile32_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_14, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_14, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_15(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

uint64_t
    * Kuiper_GEMM_Tiled_matmul_u64_tile32_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_15, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_15, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_16(uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

float
*Kuiper_GEMM_Tiled_matmul_f32_tile16_rrr(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, float *a, float *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_16, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_16, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_17(uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

double
*Kuiper_GEMM_Tiled_matmul_f64_tile16_rrr(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, double *a, double *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_17, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_17, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_18(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

uint32_t
    * Kuiper_GEMM_Tiled_matmul_u32_tile16_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_18, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_18, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_19(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

uint64_t
    * Kuiper_GEMM_Tiled_matmul_u64_tile16_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_19, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_19, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
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
__hoisted_20(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

float
*Kuiper_GEMM_Tiled_matmul_f32_tile16_ccc(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, float *a, float *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_20, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_20, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_21(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

double
*Kuiper_GEMM_Tiled_matmul_f64_tile16_ccc(uint32_t rows,
                                         uint32_t shared,
                                         uint32_t cols, double *a, double *b)
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_21, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_21, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_22(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

uint32_t
    * Kuiper_GEMM_Tiled_matmul_u32_tile16_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_22, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_22, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
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
__hoisted_23(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

uint64_t
    * Kuiper_GEMM_Tiled_matmul_u64_tile16_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_23, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_23, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
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
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_24(uint32_t tile,
             uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_rrr(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   float *gA, float *gB, float *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_24, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_24, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_25(uint32_t tile,
             uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_rrr(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   double *gA, double *gB, double *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_25, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_25, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_26(uint32_t tile,
             uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_rrr(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_26, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_26, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_27(uint32_t tile,
             uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < tile; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared +
                   vbk * tile + k] * tB[(vbk * tile + k) * cols +
                                        blockIdx.x % mcols * tile +
                                        threadIdx.x % tile];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
          blockIdx.x % mcols * tile + threadIdx.x % tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_rrr(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_27, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_27, mrows * mcols, tile * tile, 0U, tile, shared, cols,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void
__hoisted_28(uint32_t tile,
             uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_ccc(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   float *gA, float *gB, float *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_28, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_28, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void
__hoisted_29(uint32_t tile,
             uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_ccc(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   double *gA, double *gB, double *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_29, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_29, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_ccc
*/
static void
__hoisted_30(uint32_t tile,
             uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_ccc(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_30, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_30, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_ccc
*/
static void
__hoisted_31(uint32_t tile,
             uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < tile; k++)
            sum1 +=
                tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile +
                   threadIdx.x / tile] * tB[(blockIdx.x % mcols * tile +
                                             threadIdx.x % tile) * shared +
                                            vbk * tile + k];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
          blockIdx.x / mcols * tile + threadIdx.x / tile]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_ccc(uint32_t tile,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    KPR_ASSERT(tile > 0U);
    KPR_GUARD(rows % tile == 0U);
    KPR_GUARD(shared % tile == 0U);
    KPR_GUARD(cols % tile == 0U);
    uint32_t mrows = rows / tile;
    uint32_t mshared = shared / tile;
    uint32_t mcols = cols / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_31, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_31, mrows * mcols, tile * tile, 0U, tile, rows, shared,
              gA, gB, gC, mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_32(uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_32, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_32, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_33(uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_rrr(uint32_t rows,
                                          uint32_t shared,
                                          uint32_t cols,
                                          double *gA, double *gB, double *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_33, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_33, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_34(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_34, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_34, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_35(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 32U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * shared +
                   vbk * 32U + k] * tB[(vbk * 32U + k) * cols +
                                       blockIdx.x % mcols * 32U +
                                       threadIdx.x % 32U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 32U + threadIdx.x / 32U) * cols +
          blockIdx.x % mcols * 32U + threadIdx.x % 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_35, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_35, mrows * mcols, 1024U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_ccc
*/
static void
__hoisted_36(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_36, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_36, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_ccc
*/
static void
__hoisted_37(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_ccc(uint32_t rows,
                                          uint32_t shared,
                                          uint32_t cols,
                                          double *gA, double *gB, double *gC)
{
    KPR_GUARD(rows % 32U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 32U == 0U);
    uint32_t mrows = rows / 32U;
    uint32_t mshared = shared / 32U;
    uint32_t mcols = cols / 32U;
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_37, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_37, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_ccc
*/
static void
__hoisted_38(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_38, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_38, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_ccc
*/
static void
__hoisted_39(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 32U; k++)
            sum1 +=
                tA[(vbk * 32U + k) * rows + blockIdx.x / mcols * 32U +
                   threadIdx.x / 32U] * tB[(blockIdx.x % mcols * 32U +
                                            threadIdx.x % 32U) * shared +
                                           vbk * 32U + k];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 32U + threadIdx.x % 32U) * rows +
          blockIdx.x / mcols * 32U + threadIdx.x / 32U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_39, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_39, mrows * mcols, 1024U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_40(uint32_t shared,
             uint32_t cols,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_40, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_40, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_41(uint32_t shared,
             uint32_t cols,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_rrr(uint32_t rows,
                                          uint32_t shared,
                                          uint32_t cols,
                                          double *gA, double *gB, double *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_41, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_41, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_42(uint32_t shared,
             uint32_t cols,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_42, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_42, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_43(uint32_t shared,
             uint32_t cols,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 16U; k++)
            sum1 +=
                tA[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * shared +
                   vbk * 16U + k] * tB[(vbk * 16U + k) * cols +
                                       blockIdx.x % mcols * 16U +
                                       threadIdx.x % 16U];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x / mcols * 16U + threadIdx.x / 16U) * cols +
          blockIdx.x % mcols * 16U + threadIdx.x % 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_rrr(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_43, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_43, mrows * mcols, 256U, 0U, shared, cols, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_ccc
*/
static void
__hoisted_44(uint32_t rows,
             uint32_t shared,
             float *gA, float *gB, float *gC, uint32_t mshared, uint32_t mcols)
{
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        float *tA = gA;
        float *tB = gB;
        uint32_t k = 0U;
        float sum1 = 0.0f;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        float s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_44, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_44, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_ccc
*/
static void
__hoisted_45(uint32_t rows,
             uint32_t shared,
             double *gA,
             double *gB, double *gC, uint32_t mshared, uint32_t mcols)
{
    double *gTile = gC;
    double sum = 0.0l;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        double *tA = gA;
        double *tB = gB;
        uint32_t k = 0U;
        double sum1 = 0.0l;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        double s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_ccc(uint32_t rows,
                                          uint32_t shared,
                                          uint32_t cols,
                                          double *gA, double *gB, double *gC)
{
    KPR_GUARD(rows % 16U == 0U);
    KPR_GUARD(shared % 16U == 0U);
    KPR_GUARD(cols % 16U == 0U);
    uint32_t mrows = rows / 16U;
    uint32_t mshared = shared / 16U;
    uint32_t mcols = cols / 16U;
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_45, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_45, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_ccc
*/
static void
__hoisted_46(uint32_t rows,
             uint32_t shared,
             uint32_t *gA,
             uint32_t *gB, uint32_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint32_t *tA = gA;
        uint32_t *tB = gB;
        uint32_t k = 0U;
        uint32_t sum1 = 0U;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        uint32_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_46, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_46, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_ccc
*/
static void
__hoisted_47(uint32_t rows,
             uint32_t shared,
             uint64_t *gA,
             uint64_t *gB, uint64_t *gC, uint32_t mshared, uint32_t mcols)
{
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < mshared; bk++) {
        uint32_t vbk = bk;
        uint64_t *tA = gA;
        uint64_t *tB = gB;
        uint32_t k = 0U;
        uint64_t sum1 = 0ULL;
        for (; k < 16U; k++)
            sum1 +=
                tA[(vbk * 16U + k) * rows + blockIdx.x / mcols * 16U +
                   threadIdx.x / 16U] * tB[(blockIdx.x % mcols * 16U +
                                            threadIdx.x % 16U) * shared +
                                           vbk * 16U + k];
        uint64_t s_ = sum1;
        sum += s_;
    }
    gTile[(blockIdx.x % mcols * 16U + threadIdx.x % 16U) * rows +
          blockIdx.x / mcols * 16U + threadIdx.x / 16U]
        = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_ccc(uint32_t rows,
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
    KPR_SHMEM_FITS(0U);
    MUST(cudaFuncSetAttribute
         (__hoisted_47, cudaFuncAttributeMaxDynamicSharedMemorySize, 0U));
    KPR_KCALL(__hoisted_47, mrows * mcols, 256U, 0U, rows, shared, gA, gB, gC,
              mshared, mcols);
    MUST(cudaDeviceSynchronize());
}
