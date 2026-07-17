
#include "Klas_GEMM_SHMem.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_g_matmul_f32_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             float *gA,
                             float *gB, float *gC, uint32_t nn, uint32_t kk)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4U * tile * tile);
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float
         v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        float
         v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        float t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_f32_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(4U * tile * tile + 4U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * tile * tile + 4U * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0,
              mm * nn,
              tile * tile,
              4U * tile * tile + 4U * tile * tile,
              tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_g_matmul_f64_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             double *gA,
                             double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8U * tile * tile);
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double
         v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        double
         v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        double t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_f64_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(8U * tile * tile + 8U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * tile * tile + 8U * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0,
              mm * nn,
              tile * tile,
              8U * tile * tile + 8U * tile * tile,
              tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_g_matmul_u32_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             uint32_t *gA,
                             uint32_t *gB,
                             uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4U * tile * tile);
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t
            v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        uint32_t v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        uint32_t t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_u32_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(4U * tile * tile + 4U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * tile * tile + 4U * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_u32_rrr_0,
              mm * nn,
              tile * tile,
              4U * tile * tile + 4U * tile * tile,
              tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_g_matmul_u64_rrr_0(uint32_t tile,
                             uint32_t n,
                             uint32_t k,
                             uint64_t *gA,
                             uint64_t *gB,
                             uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8U * tile * tile);
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t
            v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        uint64_t v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        uint64_t t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_u64_rrr(uint32_t tile,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(8U * tile * tile + 8U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * tile * tile + 8U * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_u64_rrr_0,
              mm * nn,
              tile * tile,
              8U * tile * tile + 8U * tile * tile,
              tile, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

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
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float
         v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        float
         v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        float t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_f32_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_matmul_f32_tile32_rrr_0,
              mm * nn, 1024U, 8192U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double
         v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        double
         v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        double t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_f64_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_matmul_f64_tile32_rrr_0,
              mm * nn, 1024U, 16384U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t
            v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        uint32_t v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        uint32_t t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_u32_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t *gA,
                                        uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_matmul_u32_tile32_rrr_0,
              mm * nn, 1024U, 8192U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t
            v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        uint64_t v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        uint64_t t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_u64_tile32_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint64_t *gA,
                                        uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_matmul_u64_tile32_rrr_0,
              mm * nn, 1024U, 16384U, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float
         v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        float
         v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        float t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_f32_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_matmul_f32_tile16_rrr_0, mm * nn, 256U, 2048U, n, k,
              gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double
         v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        double
         v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        double t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_f64_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_matmul_f64_tile16_rrr_0, mm * nn, 256U, 4096U, n, k,
              gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t
            v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        uint32_t v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        uint32_t t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_u32_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t *gA,
                                        uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_matmul_u32_tile16_rrr_0, mm * nn, 256U, 2048U, n, k,
              gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t
            v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        uint64_t v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        uint64_t t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = sum;
}

void
Klas_GEMM_SHMem_g_matmul_u64_tile16_rrr(uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint64_t *gA,
                                        uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_matmul_u64_tile16_rrr_0, mm * nn, 256U, 4096U, n, k,
              gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_rrr
*/
static void
__hoisted_g_gemm_f32_rrr_0(uint32_t tile,
                           float alpha,
                           float beta,
                           uint32_t n,
                           uint32_t k,
                           float *gA,
                           float *gB, float *gC, uint32_t nn, uint32_t kk)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float *sa2 = (float *)KPR_SHMEM_AT(4U * tile * tile);
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float
         v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        float
         v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        float t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = beta * gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
                       tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_f32_rrr(uint32_t tile,
                               float alpha,
                               float beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(4U * tile * tile + 4U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * tile * tile + 4U * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_f32_rrr_0,
              mm * nn,
              tile * tile,
              4U * tile * tile + 4U * tile * tile,
              tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f64_rrr
*/
static void
__hoisted_g_gemm_f64_rrr_0(uint32_t tile,
                           double alpha,
                           double beta,
                           uint32_t n,
                           uint32_t k,
                           double *gA,
                           double *gB, double *gC, uint32_t nn, uint32_t kk)
{
    double *sa1 = (double *)KPR_SHMEM_AT(0U);
    double *sa2 = (double *)KPR_SHMEM_AT(8U * tile * tile);
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double
         v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        double
         v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        double t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = beta * gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
                       tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_f64_rrr(uint32_t tile,
                               double alpha,
                               double beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(8U * tile * tile + 8U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * tile * tile + 8U * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_f64_rrr_0,
              mm * nn,
              tile * tile,
              8U * tile * tile + 8U * tile * tile,
              tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u32_rrr
*/
static void
__hoisted_g_gemm_u32_rrr_0(uint32_t tile,
                           uint32_t alpha,
                           uint32_t beta,
                           uint32_t n,
                           uint32_t k,
                           uint32_t *gA,
                           uint32_t *gB, uint32_t *gC, uint32_t nn, uint32_t kk)
{
    uint32_t *sa1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *sa2 = (uint32_t *) KPR_SHMEM_AT(4U * tile * tile);
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t
            v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        uint32_t v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        uint32_t t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = beta * gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
                       tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_u32_rrr(uint32_t tile,
                               uint32_t alpha,
                               uint32_t beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k,
                               uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(4U * tile * tile + 4U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * tile * tile + 4U * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_u32_rrr_0,
              mm * nn,
              tile * tile,
              4U * tile * tile + 4U * tile * tile,
              tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_u64_rrr
*/
static void
__hoisted_g_gemm_u64_rrr_0(uint32_t tile,
                           uint64_t alpha,
                           uint64_t beta,
                           uint32_t n,
                           uint32_t k,
                           uint64_t *gA,
                           uint64_t *gB, uint64_t *gC, uint32_t nn, uint32_t kk)
{
    uint64_t *sa1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *sa2 = (uint64_t *) KPR_SHMEM_AT(8U * tile * tile);
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t
            v1 =
            gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k + tile * bk +
               threadIdx.x % tile];
        uint64_t v2 =
            gB[(tile * bk + threadIdx.x / tile) * n + tile * (blockIdx.x % nn) +
               threadIdx.x % tile];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / tile * tile + vk] * sa2[vk * tile +
                                                          threadIdx.x % tile];
        }
        uint64_t t = sum1;
        sum += t;
    }
    gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
          tile * (blockIdx.x % nn) + threadIdx.x % tile]
        = beta * gTile[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * n +
                       tile * (blockIdx.x % nn) + threadIdx.x % tile]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_u64_rrr(uint32_t tile,
                               uint64_t alpha,
                               uint64_t beta,
                               uint32_t m,
                               uint32_t n,
                               uint32_t k,
                               uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    KPR_SHMEM_FITS(8U * tile * tile + 8U * tile * tile);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * tile * tile + 8U * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_u64_rrr_0,
              mm * nn,
              tile * tile,
              8U * tile * tile + 8U * tile * tile,
              tile, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float
         v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        float
         v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        float t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = beta * gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
                       32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_f32_tile32_rrr(float alpha,
                                      float beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_f32_tile32_rrr_0,
              mm * nn, 1024U, 8192U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double
         v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        double
         v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        double t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = beta * gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
                       32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_f64_tile32_rrr(double alpha,
                                      double beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_f64_tile32_rrr_0,
              mm * nn, 1024U, 16384U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t
            v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        uint32_t v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        uint32_t t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = beta * gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
                       32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                      uint32_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_g_gemm_u32_tile32_rrr_0,
              mm * nn, 1024U, 8192U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t
            v1 =
            gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k + 32U * bk +
               threadIdx.x % 32U];
        uint64_t v2 =
            gB[(32U * bk + threadIdx.x / 32U) * n + 32U * (blockIdx.x % nn) +
               threadIdx.x % 32U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 32U * 32U + vk] * sa2[vk * 32U +
                                                        threadIdx.x % 32U];
        }
        uint64_t t = sum1;
        sum += t;
    }
    gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
          32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        = beta * gTile[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * n +
                       32U * (blockIdx.x % nn) + threadIdx.x % 32U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                      uint64_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_tile32_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_g_gemm_u64_tile32_rrr_0,
              mm * nn, 1024U, 16384U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    float *gTile = gC;
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float
         v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        float
         v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        float t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = beta * gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
                       16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_f32_tile16_rrr(float alpha,
                                      float beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_gemm_f32_tile16_rrr_0,
              mm * nn, 256U, 2048U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    double *gTile = gC;
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double
         v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        double
         v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        double t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = beta * gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
                       16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_f64_tile16_rrr(double alpha,
                                      double beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_f64_tile16_rrr_0,
              mm * nn, 256U, 4096U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint32_t *gTile = gC;
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t
            v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        uint32_t v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        uint32_t t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = beta * gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
                       16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                      uint32_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_g_gemm_u32_tile16_rrr_0,
              mm * nn, 256U, 2048U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
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
    uint64_t *gTile = gC;
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t
            v1 =
            gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k + 16U * bk +
               threadIdx.x % 16U];
        uint64_t v2 =
            gB[(16U * bk + threadIdx.x / 16U) * n + 16U * (blockIdx.x % nn) +
               threadIdx.x % 16U];
        __syncthreads();
        sa1[threadIdx.x] = v1;
        sa2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 +=
                sa1[threadIdx.x / 16U * 16U + vk] * sa2[vk * 16U +
                                                        threadIdx.x % 16U];
        }
        uint64_t t = sum1;
        sum += t;
    }
    gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
          16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        = beta * gTile[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * n +
                       16U * (blockIdx.x % nn) + threadIdx.x % 16U]
        + alpha * sum;
}

void
Klas_GEMM_SHMem_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                      uint64_t beta,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_tile16_rrr_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_g_gemm_u64_tile16_rrr_0,
              mm * nn, 256U, 4096U, alpha, beta, n, k, gA, gB, gC, nn, kk);
    MUST(cudaDeviceSynchronize());
}
