
#include "Klas_Reduce.h"

typedef struct __uint32_t__uint32_t_______s {
    uint32_t fst;
    uint32_t snd;
} __uint32_t__uint32_t______;

__global__
/**
  hoisted when extracting mean_fw_f32_row
*/
static void __hoisted_mean_fw_f32_row_0(uint32_t cols, float *x, float *y)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < cols; idx += 1024U) {
        __uint32_t__uint32_t______ scrut = {.fst = blockIdx.x,.snd = idx };
        acc += x[scrut.fst * cols + scrut.snd];
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        y[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting mean_fw_f32_row
*/
static void __hoisted_mean_fw_f32_row_1(uint32_t rows, float inv_cols, float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < rows)
        y[1024U * blockIdx.x + threadIdx.x] *= inv_cols;
}

void
Klas_Reduce_mean_fw_f32_row(uint32_t rows, uint32_t cols, float inv_cols,
                            float *x, float *y)
{
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_mean_fw_f32_row_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_mean_fw_f32_row_0, rows, 1024U, 4096U, cols, x, y);
    MUST(cudaDeviceSynchronize());
    KPR_KCALL(__hoisted_mean_fw_f32_row_1,
              rows / 1024U + (uint32_t) (rows % 1024U != 0U),
              1024U, 0U, rows, inv_cols, y);
    MUST(cudaDeviceSynchronize());
}
