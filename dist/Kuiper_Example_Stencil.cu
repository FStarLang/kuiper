
#include "Kuiper_Example_Stencil.h"

__global__
/**
  hoisted when extracting stencil3x3_f32_add_rr
*/
static void __hoisted_0(float *gIn, float *gOut, uint32_t cols_sub2)
{
    uint32_t i = blockIdx.x / cols_sub2;
    uint32_t j = blockIdx.x % cols_sub2;
    gOut[i * cols_sub2 + j] =
        gIn[i * (cols_sub2 + 2U) + j] * 1.0f + gIn[i * (cols_sub2 + 2U) + j +
                                                   1U] * 1.0f +
        gIn[i * (cols_sub2 + 2U) + j + 2U] * 1.0f +
        gIn[(i + 1U) * (cols_sub2 + 2U) + j] * 1.0f +
        gIn[(i + 1U) * (cols_sub2 + 2U) + j + 1U] * 1.0f +
        gIn[(i + 1U) * (cols_sub2 + 2U) + j + 2U] * 1.0f +
        gIn[(i + 2U) * (cols_sub2 + 2U) + j] * 1.0f +
        gIn[(i + 2U) * (cols_sub2 + 2U) + j + 1U] * 1.0f +
        gIn[(i + 2U) * (cols_sub2 + 2U) + j + 2U] * 1.0f;
}

void
Kuiper_Example_Stencil_stencil3x3_f32_add_rr(uint32_t rows,
                                             uint32_t cols,
                                             float *gIn, float *gOut)
{
    uint32_t cols_sub2 = cols - 2U;
    KPR_KCALL(__hoisted_0, (rows - 2U) * cols_sub2, 1U, 0U, gIn, gOut,
              cols_sub2);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting stencil3x3_i32_add_mul2_rc
*/
static void __hoisted_1(uint32_t *gIn, uint32_t *gOut, uint32_t rows_sub2,
                        uint32_t cols_sub2)
{
    uint32_t i = blockIdx.x / cols_sub2;
    uint32_t j = blockIdx.x % cols_sub2;
    gOut[j * rows_sub2 + i] =
        gIn[i * (cols_sub2 + 2U) + j] + gIn[i * (cols_sub2 + 2U) + j + 1U] +
        gIn[i * (cols_sub2 + 2U) + j + 2U]
        + gIn[(i + 1U) * (cols_sub2 + 2U) + j]
        + gIn[(i + 1U) * (cols_sub2 + 2U) + j + 1U] * 8U
        + gIn[(i + 1U) * (cols_sub2 + 2U) + j + 2U]
        + gIn[(i + 2U) * (cols_sub2 + 2U) + j]
        + gIn[(i + 2U) * (cols_sub2 + 2U) + j + 1U]
        + gIn[(i + 2U) * (cols_sub2 + 2U) + j + 2U];
}

void
Kuiper_Example_Stencil_stencil3x3_i32_add_mul2_rc(uint32_t rows,
                                                  uint32_t cols,
                                                  uint32_t *gIn, uint32_t *gOut)
{
    uint32_t rows_sub2 = rows - 2U;
    uint32_t cols_sub2 = cols - 2U;
    KPR_KCALL(__hoisted_1, rows_sub2 * cols_sub2, 1U, 0U, gIn, gOut, rows_sub2,
              cols_sub2);
    MUST(cudaDeviceSynchronize());
}
