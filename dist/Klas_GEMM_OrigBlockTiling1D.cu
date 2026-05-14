
#include "Klas_GEMM_OrigBlockTiling1D.h"

__global__
/**
  hoisted when extracting matmul_f32_tiles64x8_8x64_rc8_rrr
*/
static void __hoisted_0(uint32_t shared, uint32_t cols, float *gA, float *gB,
                        float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(2048U);
    uint32_t mrow = blockIdx.x / (cols / 64U);
    uint32_t mcol = blockIdx.x % (cols / 64U);
    float cache1d[8U];
    memset(cache1d, 0U, 8U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 8U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        sA[threadIdx.x] =
            gA[(mrow * 64U + threadIdx.x / 8U) * shared + __anf01 * 8U +
               threadIdx.x % 8U];
        sB[threadIdx.x] =
            gB[(__anf01 * 8U + threadIdx.x / 64U) * cols + mcol * 64U +
               threadIdx.x % 64U];
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 8U; dotIdx++) {
            uint32_t resIdx = 0U;
            for (; resIdx < 8U; resIdx++)
                cache1d[resIdx] +=
                    sA[(threadIdx.x / 64U * 8U + resIdx) * 8U +
                       dotIdx] * sB[dotIdx * 64U + threadIdx.x % 64U];
        }
    }
    uint32_t resIdx = 0U;
    for (; resIdx < 8U; resIdx++)
        gC[(mrow * 64U + threadIdx.x / 64U * 8U + resIdx) * cols + mcol * 64U +
           threadIdx.x % 64U] = cache1d[resIdx];
}

void
Klas_GEMM_OrigBlockTiling1D_matmul_f32_tiles64x8_8x64_rc8_rrr(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              float *gA,
                                                              float *gB,
                                                              float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 8U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_0, rows / 64U * (cols / 64U), 512U, 4096U, shared, cols,
              gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting g_gemm_f32_tiles64x8_8x64_rc8_rrr
*/
static void
__hoisted_1(float alpha,
            float beta,
            uint32_t shared, uint32_t cols, float *gA, float *gB, float *gC)
{
    float *sA = (float *)KPR_SHMEM_AT(0U);
    float *sB = (float *)KPR_SHMEM_AT(2048U);
    uint32_t mrow = blockIdx.x / (cols / 64U);
    uint32_t mcol = blockIdx.x % (cols / 64U);
    float cache1d[8U];
    memset(cache1d, 0U, 8U * sizeof(float));
    uint32_t bkIdx = 0U;
    for (; bkIdx < shared / 8U; bkIdx++) {
        __syncthreads();
        uint32_t __anf01 = bkIdx;
        sA[threadIdx.x] =
            gA[(mrow * 64U + threadIdx.x / 8U) * shared + __anf01 * 8U +
               threadIdx.x % 8U];
        sB[threadIdx.x] =
            gB[(__anf01 * 8U + threadIdx.x / 64U) * cols + mcol * 64U +
               threadIdx.x % 64U];
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 8U; dotIdx++) {
            uint32_t resIdx = 0U;
            for (; resIdx < 8U; resIdx++)
                cache1d[resIdx] +=
                    sA[(threadIdx.x / 64U * 8U + resIdx) * 8U +
                       dotIdx] * sB[dotIdx * 64U + threadIdx.x % 64U];
        }
    }
    uint32_t resIdx = 0U;
    for (; resIdx < 8U; resIdx++) {
        float *tC = gC;
        tC[(mrow * 64U + threadIdx.x / 64U * 8U + resIdx) * cols + mcol * 64U +
           threadIdx.x % 64U] =
beta * tC[(mrow * 64U + threadIdx.x / 64U * 8U + resIdx) * cols + mcol * 64U + threadIdx.x % 64U]
            + alpha * cache1d[resIdx];
    }
}

void
Klas_GEMM_OrigBlockTiling1D_g_gemm_f32_tiles64x8_8x64_rc8_rrr(float alpha,
                                                              float beta,
                                                              uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              float *gA,
                                                              float *gB,
                                                              float *gC)
{
    KPR_GUARD(rows % 64U == 0U);
    KPR_GUARD(shared % 8U == 0U);
    KPR_GUARD(cols % 64U == 0U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096U));
    KPR_KCALL(__hoisted_1, rows / 64U * (cols / 64U), 512U, 4096U, alpha, beta,
              shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}
