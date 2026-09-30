
#include "Klas_GEMM_SHMem.h"

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_g_matmul_f32_rrr_0(uint32_t tile, float *gA, float *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, float *gC)
{
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * tile * tile);
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                      tile * bk + threadIdx.x % tile];
        float v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                      tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * tile + threadIdx.x / tile) * n +
        blockIdx.x % nn * tile + threadIdx.x % tile] = sum;
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_g_matmul_f64_rrr_0(uint32_t tile, double *gA, double *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, double *gC)
{
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 =
        (double *) KPR_SHMEM_AT((uint32_t) sizeof(double) * tile * tile);
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                       tile * bk + threadIdx.x % tile];
        double v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                       tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * tile + threadIdx.x / tile) * n +
        blockIdx.x % nn * tile + threadIdx.x % tile] = sum;
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_g_matmul_u32_rrr_0(uint32_t tile, uint32_t *gA, uint32_t *gB,
    uint32_t kk, uint32_t nn, uint32_t k, uint32_t n, uint32_t *gC)
{
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * tile * tile);
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                         tile * bk + threadIdx.x % tile];
        uint32_t v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                         tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * tile + threadIdx.x / tile) * n +
        blockIdx.x % nn * tile + threadIdx.x % tile] = sum;
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_g_matmul_u64_rrr_0(uint32_t tile, uint64_t *gA, uint64_t *gB,
    uint32_t kk, uint32_t nn, uint32_t k, uint32_t n, uint64_t *gC)
{
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 =
        (uint64_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint64_t) * tile * tile);
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                         tile * bk + threadIdx.x % tile];
        uint64_t v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                         tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * tile + threadIdx.x / tile) * n +
        blockIdx.x % nn * tile + threadIdx.x % tile] = sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_g_matmul_f32_tile32_rrr_0(float *gA, float *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, float *gC)
{
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                      32U * bk + threadIdx.x % 32U];
        float v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                      32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 32U + threadIdx.x / 32U) * n + blockIdx.x % nn * 32U +
        threadIdx.x % 32U] = sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_g_matmul_f64_tile32_rrr_0(double *gA, double *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, double *gC)
{
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT((uint32_t) sizeof(double) * 1024U);
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                       32U * bk + threadIdx.x % 32U];
        double v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                       32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 32U + threadIdx.x / 32U) * n + blockIdx.x % nn * 32U +
        threadIdx.x % 32U] = sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_g_matmul_u32_tile32_rrr_0(uint32_t *gA, uint32_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint32_t *gC)
{
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * 1024U);
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                         32U * bk + threadIdx.x % 32U];
        uint32_t v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 32U + threadIdx.x / 32U) * n + blockIdx.x % nn * 32U +
        threadIdx.x % 32U] = sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_g_matmul_u64_tile32_rrr_0(uint64_t *gA, uint64_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint64_t *gC)
{
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 =
        (uint64_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint64_t) * 1024U);
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                         32U * bk + threadIdx.x % 32U];
        uint64_t v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 32U + threadIdx.x / 32U) * n + blockIdx.x % nn * 32U +
        threadIdx.x % 32U] = sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_g_matmul_f32_tile16_rrr_0(float *gA, float *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, float *gC)
{
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                      16U * bk + threadIdx.x % 16U];
        float v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                      16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 16U + threadIdx.x / 16U) * n + blockIdx.x % nn * 16U +
        threadIdx.x % 16U] = sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_g_matmul_f64_tile16_rrr_0(double *gA, double *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, double *gC)
{
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT((uint32_t) sizeof(double) * 256U);
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                       16U * bk + threadIdx.x % 16U];
        double v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                       16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 16U + threadIdx.x / 16U) * n + blockIdx.x % nn * 16U +
        threadIdx.x % 16U] = sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_g_matmul_u32_tile16_rrr_0(uint32_t *gA, uint32_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint32_t *gC)
{
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * 256U);
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                         16U * bk + threadIdx.x % 16U];
        uint32_t v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 16U + threadIdx.x / 16U) * n + blockIdx.x % nn * 16U +
        threadIdx.x % 16U] = sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_g_matmul_u64_tile16_rrr_0(uint64_t *gA, uint64_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint64_t *gC)
{
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 =
        (uint64_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint64_t) * 256U);
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                         16U * bk + threadIdx.x % 16U];
        uint64_t v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    gC[(blockIdx.x / nn * 16U + threadIdx.x / 16U) * n + blockIdx.x % nn * 16U +
        threadIdx.x % 16U] = sum;
}

__global__
/**
  hoisted when extracting g_gemm_f32_rrr
*/
static void
__hoisted_g_gemm_f32_rrr_0(uint32_t tile, float *gA, float *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, float *gC, float beta, float alpha)
{
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * tile * tile);
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                      tile * bk + threadIdx.x % tile];
        float v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                      tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * tile + threadIdx.x / tile;
    uint32_t gcol_sz = blockIdx.x % nn * tile + threadIdx.x % tile;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__
/**
  hoisted when extracting g_gemm_f64_rrr
*/
static void
__hoisted_g_gemm_f64_rrr_0(uint32_t tile, double *gA, double *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, double *gC, double beta, double alpha)
{
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 =
        (double *) KPR_SHMEM_AT((uint32_t) sizeof(double) * tile * tile);
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                       tile * bk + threadIdx.x % tile];
        double v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                       tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * tile + threadIdx.x / tile;
    uint32_t gcol_sz = blockIdx.x % nn * tile + threadIdx.x % tile;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__
/**
  hoisted when extracting g_gemm_u32_rrr
*/
static void
__hoisted_g_gemm_u32_rrr_0(uint32_t tile, uint32_t *gA, uint32_t *gB,
    uint32_t kk, uint32_t nn, uint32_t k, uint32_t n, uint32_t *gC,
    uint32_t beta, uint32_t alpha)
{
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * tile * tile);
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                         tile * bk + threadIdx.x % tile];
        uint32_t v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                         tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * tile + threadIdx.x / tile;
    uint32_t gcol_sz = blockIdx.x % nn * tile + threadIdx.x % tile;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__
/**
  hoisted when extracting g_gemm_u64_rrr
*/
static void
__hoisted_g_gemm_u64_rrr_0(uint32_t tile, uint64_t *gA, uint64_t *gB,
    uint32_t kk, uint32_t nn, uint32_t k, uint32_t n, uint64_t *gC,
    uint64_t beta, uint64_t alpha)
{
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 =
        (uint64_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint64_t) * tile * tile);
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t v1 = gA[(tile * (blockIdx.x / nn) + threadIdx.x / tile) * k +
                         tile * bk + threadIdx.x % tile];
        uint64_t v2 = gB[(tile * bk + threadIdx.x / tile) * n +
                         tile * (blockIdx.x % nn) + threadIdx.x % tile];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < tile; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / tile * tile + vk] *
                    ar2[vk * tile + threadIdx.x % tile];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * tile + threadIdx.x / tile;
    uint32_t gcol_sz = blockIdx.x % nn * tile + threadIdx.x % tile;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_f32_tile32_rrr
*/
static void
__hoisted_g_gemm_f32_tile32_rrr_0(float *gA, float *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, float *gC, float beta, float alpha)
{
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 1024U);
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                      32U * bk + threadIdx.x % 32U];
        float v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                      32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 32U + threadIdx.x / 32U;
    uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x % 32U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_f64_tile32_rrr
*/
static void
__hoisted_g_gemm_f64_tile32_rrr_0(double *gA, double *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, double *gC, double beta, double alpha)
{
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT((uint32_t) sizeof(double) * 1024U);
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                       32U * bk + threadIdx.x % 32U];
        double v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                       32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 32U + threadIdx.x / 32U;
    uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x % 32U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_u32_tile32_rrr
*/
static void
__hoisted_g_gemm_u32_tile32_rrr_0(uint32_t *gA, uint32_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint32_t *gC, uint32_t beta,
    uint32_t alpha)
{
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * 1024U);
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                         32U * bk + threadIdx.x % 32U];
        uint32_t v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 32U + threadIdx.x / 32U;
    uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x % 32U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting g_gemm_u64_tile32_rrr
*/
static void
__hoisted_g_gemm_u64_tile32_rrr_0(uint64_t *gA, uint64_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint64_t *gC, uint64_t beta,
    uint64_t alpha)
{
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 =
        (uint64_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint64_t) * 1024U);
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t v1 = gA[(32U * (blockIdx.x / nn) + threadIdx.x / 32U) * k +
                         32U * bk + threadIdx.x % 32U];
        uint64_t v2 = gB[(32U * bk + threadIdx.x / 32U) * n +
                         32U * (blockIdx.x % nn) + threadIdx.x % 32U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 32U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 32U * 32U + vk] *
                    ar2[vk * 32U + threadIdx.x % 32U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 32U + threadIdx.x / 32U;
    uint32_t gcol_sz = blockIdx.x % nn * 32U + threadIdx.x % 32U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_f32_tile16_rrr
*/
static void
__hoisted_g_gemm_f32_tile16_rrr_0(float *gA, float *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, float *gC, float beta, float alpha)
{
    float *ar1 = (float *) KPR_SHMEM_AT(0U);
    float *ar2 = (float *) KPR_SHMEM_AT((uint32_t) sizeof(float) * 256U);
    float sum = 0.0f;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        float v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                      16U * bk + threadIdx.x % 16U];
        float v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                      16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        float sum1 = 0.0f;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 16U + threadIdx.x / 16U;
    uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x % 16U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_f64_tile16_rrr
*/
static void
__hoisted_g_gemm_f64_tile16_rrr_0(double *gA, double *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, double *gC, double beta, double alpha)
{
    double *ar1 = (double *) KPR_SHMEM_AT(0U);
    double *ar2 = (double *) KPR_SHMEM_AT((uint32_t) sizeof(double) * 256U);
    double sum = 0.0;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        double v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                       16U * bk + threadIdx.x % 16U];
        double v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                       16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        double sum1 = 0.0;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 16U + threadIdx.x / 16U;
    uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x % 16U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_u32_tile16_rrr
*/
static void
__hoisted_g_gemm_u32_tile16_rrr_0(uint32_t *gA, uint32_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint32_t *gC, uint32_t beta,
    uint32_t alpha)
{
    uint32_t *ar1 = (uint32_t *) KPR_SHMEM_AT(0U);
    uint32_t *ar2 =
        (uint32_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint32_t) * 256U);
    uint32_t sum = 0U;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint32_t v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                         16U * bk + threadIdx.x % 16U];
        uint32_t v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint32_t sum1 = 0U;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 16U + threadIdx.x / 16U;
    uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x % 16U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

__global__ __launch_bounds__(256)
/**
  hoisted when extracting g_gemm_u64_tile16_rrr
*/
static void
__hoisted_g_gemm_u64_tile16_rrr_0(uint64_t *gA, uint64_t *gB, uint32_t kk,
    uint32_t nn, uint32_t k, uint32_t n, uint64_t *gC, uint64_t beta,
    uint64_t alpha)
{
    uint64_t *ar1 = (uint64_t *) KPR_SHMEM_AT(0U);
    uint64_t *ar2 =
        (uint64_t *) KPR_SHMEM_AT((uint32_t) sizeof(uint64_t) * 256U);
    uint64_t sum = 0ULL;
    uint32_t bk = 0U;
    for (; bk < kk; bk++) {
        uint64_t v1 = gA[(16U * (blockIdx.x / nn) + threadIdx.x / 16U) * k +
                         16U * bk + threadIdx.x % 16U];
        uint64_t v2 = gB[(16U * bk + threadIdx.x / 16U) * n +
                         16U * (blockIdx.x % nn) + threadIdx.x % 16U];
        __syncthreads();
        ar1[threadIdx.x] = v1;
        ar2[threadIdx.x] = v2;
        __syncthreads();
        uint32_t k1 = 0U;
        uint64_t sum1 = 0ULL;
        for (; k1 < 16U; k1++) {
            uint32_t vk = k1;
            sum1 += ar1[threadIdx.x / 16U * 16U + vk] *
                    ar2[vk * 16U + threadIdx.x % 16U];
        }
        sum += sum1;
    }
    uint32_t grow_sz = blockIdx.x / nn * 16U + threadIdx.x / 16U;
    uint32_t gcol_sz = blockIdx.x % nn * 16U + threadIdx.x % 16U;
    gC[grow_sz * n + gcol_sz] = beta * gC[grow_sz * n + gcol_sz] + alpha * sum;
}

void Klas_GEMM_SHMem_g_matmul_f32_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * tile * tile +
                   (uint32_t) sizeof(float) * tile * tile);
    if ((uint32_t) sizeof(float) * tile * tile +
            (uint32_t) sizeof(float) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * tile * tile +
                (uint32_t) sizeof(float) * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_f32_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(float) * tile * tile +
            (uint32_t) sizeof(float) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_f64_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * tile * tile +
                   (uint32_t) sizeof(double) * tile * tile);
    if ((uint32_t) sizeof(double) * tile * tile +
            (uint32_t) sizeof(double) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * tile * tile +
                (uint32_t) sizeof(double) * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_f64_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(double) * tile * tile +
            (uint32_t) sizeof(double) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_u32_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * tile * tile +
                   (uint32_t) sizeof(uint32_t) * tile * tile);
    if ((uint32_t) sizeof(uint32_t) * tile * tile +
            (uint32_t) sizeof(uint32_t) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * tile * tile +
                (uint32_t) sizeof(uint32_t) * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_u32_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(uint32_t) * tile * tile +
            (uint32_t) sizeof(uint32_t) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_u64_rrr(uint32_t tile, uint32_t m, uint32_t n,
    uint32_t k, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint64_t) * tile * tile +
                   (uint32_t) sizeof(uint64_t) * tile * tile);
    if ((uint32_t) sizeof(uint64_t) * tile * tile +
            (uint32_t) sizeof(uint64_t) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint64_t) * tile * tile +
                (uint32_t) sizeof(uint64_t) * tile * tile));
    KPR_KCALL(__hoisted_g_matmul_u64_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(uint64_t) * tile * tile +
            (uint32_t) sizeof(uint64_t) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_f32_tile32_rrr(
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_matmul_f32_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_f64_tile32_rrr(
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(double) * 1024U + (uint32_t) sizeof(double) * 1024U);
    if ((uint32_t) sizeof(double) * 1024U + (uint32_t) sizeof(double) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 1024U +
                (uint32_t) sizeof(double) * 1024U));
    KPR_KCALL(__hoisted_g_matmul_f64_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(double) * 1024U + (uint32_t) sizeof(double) * 1024U,
        s, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_u32_tile32_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * 1024U +
                   (uint32_t) sizeof(uint32_t) * 1024U);
    if ((uint32_t) sizeof(uint32_t) * 1024U +
            (uint32_t) sizeof(uint32_t) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * 1024U +
                (uint32_t) sizeof(uint32_t) * 1024U));
    KPR_KCALL(__hoisted_g_matmul_u32_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(uint32_t) * 1024U +
            (uint32_t) sizeof(uint32_t) * 1024U,
        s, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_u64_tile32_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint64_t) * 1024U +
                   (uint32_t) sizeof(uint64_t) * 1024U);
    if ((uint32_t) sizeof(uint64_t) * 1024U +
            (uint32_t) sizeof(uint64_t) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint64_t) * 1024U +
                (uint32_t) sizeof(uint64_t) * 1024U));
    KPR_KCALL(__hoisted_g_matmul_u64_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(uint64_t) * 1024U +
            (uint32_t) sizeof(uint64_t) * 1024U,
        s, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_f32_tile16_rrr(
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f32_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U));
    KPR_KCALL(__hoisted_g_matmul_f32_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U, s,
        gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_f64_tile16_rrr(
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(double) * 256U + (uint32_t) sizeof(double) * 256U);
    if ((uint32_t) sizeof(double) * 256U + (uint32_t) sizeof(double) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_f64_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 256U +
                (uint32_t) sizeof(double) * 256U));
    KPR_KCALL(__hoisted_g_matmul_f64_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(double) * 256U + (uint32_t) sizeof(double) * 256U, s,
        gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_u32_tile16_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * 256U +
                   (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(uint32_t) * 256U +
            (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u32_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_matmul_u32_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(uint32_t) * 256U + (uint32_t) sizeof(uint32_t) * 256U,
        s, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_matmul_u64_tile16_rrr(uint32_t m, uint32_t n, uint32_t k,
    uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint64_t) * 256U +
                   (uint32_t) sizeof(uint64_t) * 256U);
    if ((uint32_t) sizeof(uint64_t) * 256U +
            (uint32_t) sizeof(uint64_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_matmul_u64_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint64_t) * 256U +
                (uint32_t) sizeof(uint64_t) * 256U));
    KPR_KCALL(__hoisted_g_matmul_u64_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(uint64_t) * 256U + (uint32_t) sizeof(uint64_t) * 256U,
        s, gA, gB, kk, nn, k, n, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_f32_rrr(uint32_t tile, float alpha, float beta,
    uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(float) * tile * tile +
                   (uint32_t) sizeof(float) * tile * tile);
    if ((uint32_t) sizeof(float) * tile * tile +
            (uint32_t) sizeof(float) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * tile * tile +
                (uint32_t) sizeof(float) * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_f32_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(float) * tile * tile +
            (uint32_t) sizeof(float) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_f64_rrr(uint32_t tile, double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(double) * tile * tile +
                   (uint32_t) sizeof(double) * tile * tile);
    if ((uint32_t) sizeof(double) * tile * tile +
            (uint32_t) sizeof(double) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * tile * tile +
                (uint32_t) sizeof(double) * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_f64_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(double) * tile * tile +
            (uint32_t) sizeof(double) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_u32_rrr(uint32_t tile, uint32_t alpha,
    uint32_t beta, uint32_t m, uint32_t n, uint32_t k, uint32_t *gA,
    uint32_t *gB, uint32_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * tile * tile +
                   (uint32_t) sizeof(uint32_t) * tile * tile);
    if ((uint32_t) sizeof(uint32_t) * tile * tile +
            (uint32_t) sizeof(uint32_t) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * tile * tile +
                (uint32_t) sizeof(uint32_t) * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_u32_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(uint32_t) * tile * tile +
            (uint32_t) sizeof(uint32_t) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_u64_rrr(uint32_t tile, uint64_t alpha,
    uint64_t beta, uint32_t m, uint32_t n, uint32_t k, uint64_t *gA,
    uint64_t *gB, uint64_t *gC)
{
    uint32_t mm = m / tile;
    uint32_t nn = n / tile;
    uint32_t kk = k / tile;
    KPR_ASSERT(tile > 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint64_t) * tile * tile +
                   (uint32_t) sizeof(uint64_t) * tile * tile);
    if ((uint32_t) sizeof(uint64_t) * tile * tile +
            (uint32_t) sizeof(uint64_t) * tile * tile >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint64_t) * tile * tile +
                (uint32_t) sizeof(uint64_t) * tile * tile));
    KPR_KCALL(__hoisted_g_gemm_u64_rrr_0, mm * nn, tile * tile,
        (uint32_t) sizeof(uint64_t) * tile * tile +
            (uint32_t) sizeof(uint64_t) * tile * tile,
        s, tile, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_f32_tile32_rrr(float alpha, float beta, uint32_t m,
    uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U);
    if ((uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 1024U +
                (uint32_t) sizeof(float) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f32_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(float) * 1024U + (uint32_t) sizeof(float) * 1024U, s,
        gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_f64_tile32_rrr(double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(double) * 1024U + (uint32_t) sizeof(double) * 1024U);
    if ((uint32_t) sizeof(double) * 1024U + (uint32_t) sizeof(double) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 1024U +
                (uint32_t) sizeof(double) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_f64_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(double) * 1024U + (uint32_t) sizeof(double) * 1024U,
        s, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_u32_tile32_rrr(uint32_t alpha, uint32_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
    uint32_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * 1024U +
                   (uint32_t) sizeof(uint32_t) * 1024U);
    if ((uint32_t) sizeof(uint32_t) * 1024U +
            (uint32_t) sizeof(uint32_t) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * 1024U +
                (uint32_t) sizeof(uint32_t) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_u32_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(uint32_t) * 1024U +
            (uint32_t) sizeof(uint32_t) * 1024U,
        s, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_u64_tile32_rrr(uint64_t alpha, uint64_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
    uint64_t *gC)
{
    uint32_t mm = m / 32U;
    uint32_t nn = n / 32U;
    uint32_t kk = k / 32U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint64_t) * 1024U +
                   (uint32_t) sizeof(uint64_t) * 1024U);
    if ((uint32_t) sizeof(uint64_t) * 1024U +
            (uint32_t) sizeof(uint64_t) * 1024U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_tile32_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint64_t) * 1024U +
                (uint32_t) sizeof(uint64_t) * 1024U));
    KPR_KCALL(__hoisted_g_gemm_u64_tile32_rrr_0, mm * nn, 1024U,
        (uint32_t) sizeof(uint64_t) * 1024U +
            (uint32_t) sizeof(uint64_t) * 1024U,
        s, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_f32_tile16_rrr(float alpha, float beta, uint32_t m,
    uint32_t n, uint32_t k, float *gA, float *gB, float *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U);
    if ((uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f32_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U));
    KPR_KCALL(__hoisted_g_gemm_f32_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(float) * 256U + (uint32_t) sizeof(float) * 256U, s,
        gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_f64_tile16_rrr(double alpha, double beta,
    uint32_t m, uint32_t n, uint32_t k, double *gA, double *gB, double *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(
        (uint32_t) sizeof(double) * 256U + (uint32_t) sizeof(double) * 256U);
    if ((uint32_t) sizeof(double) * 256U + (uint32_t) sizeof(double) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_f64_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(double) * 256U +
                (uint32_t) sizeof(double) * 256U));
    KPR_KCALL(__hoisted_g_gemm_f64_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(double) * 256U + (uint32_t) sizeof(double) * 256U, s,
        gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_u32_tile16_rrr(uint32_t alpha, uint32_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint32_t *gA, uint32_t *gB,
    uint32_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint32_t) * 256U +
                   (uint32_t) sizeof(uint32_t) * 256U);
    if ((uint32_t) sizeof(uint32_t) * 256U +
            (uint32_t) sizeof(uint32_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u32_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint32_t) * 256U +
                (uint32_t) sizeof(uint32_t) * 256U));
    KPR_KCALL(__hoisted_g_gemm_u32_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(uint32_t) * 256U + (uint32_t) sizeof(uint32_t) * 256U,
        s, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Klas_GEMM_SHMem_g_gemm_u64_tile16_rrr(uint64_t alpha, uint64_t beta,
    uint32_t m, uint32_t n, uint32_t k, uint64_t *gA, uint64_t *gB,
    uint64_t *gC)
{
    uint32_t mm = m / 16U;
    uint32_t nn = n / 16U;
    uint32_t kk = k / 16U;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS((uint32_t) sizeof(uint64_t) * 256U +
                   (uint32_t) sizeof(uint64_t) * 256U);
    if ((uint32_t) sizeof(uint64_t) * 256U +
            (uint32_t) sizeof(uint64_t) * 256U >=
        49152U)
        MUST(cudaFuncSetAttribute(__hoisted_g_gemm_u64_tile16_rrr_0,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (uint32_t) sizeof(uint64_t) * 256U +
                (uint32_t) sizeof(uint64_t) * 256U));
    KPR_KCALL(__hoisted_g_gemm_u64_tile16_rrr_0, mm * nn, 256U,
        (uint32_t) sizeof(uint64_t) * 256U + (uint32_t) sizeof(uint64_t) * 256U,
        s, gA, gB, kk, nn, k, n, gC, beta, alpha);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
