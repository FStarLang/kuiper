

#include "Kuiper_Stencil.h"

__global__
/**
  hoisted when extracting stencil3x3_f32_add_rr
*/
static void __hoisted_0(float_t *gIn, float_t *gOut, uint32_t cols_sub2)
{
  uint32_t i = blockIdx.x / cols_sub2;
  uint32_t j = blockIdx.x % cols_sub2;
  gOut[i * cols_sub2 + j] =
    gIn[i * (cols_sub2 + (uint32_t)2U) + j] * (float_t)1.0f +
      gIn[i * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)1U] * (float_t)1.0f
    + gIn[i * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)2U] * (float_t)1.0f
    + gIn[(i + (uint32_t)1U) * (cols_sub2 + (uint32_t)2U) + j] * (float_t)1.0f
    + gIn[(i + (uint32_t)1U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)1U] * (float_t)1.0f
    + gIn[(i + (uint32_t)1U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)2U] * (float_t)1.0f
    + gIn[(i + (uint32_t)2U) * (cols_sub2 + (uint32_t)2U) + j] * (float_t)1.0f
    + gIn[(i + (uint32_t)2U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)1U] * (float_t)1.0f
    + gIn[(i + (uint32_t)2U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)2U] * (float_t)1.0f;
}

void
Kuiper_Stencil_stencil3x3_f32_add_rr(uint32_t rows, uint32_t cols, float_t *gIn, float_t *gOut)
{
  uint32_t cols_sub2 = cols - (uint32_t)2U;
  KPR_KCALL(__hoisted_0,
    (rows - (uint32_t)2U) * cols_sub2,
    (uint32_t)1U,
    (uint32_t)0U,
    gIn,
    gOut,
    cols_sub2);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting stencil3x3_i32_add_mul2_rc
*/
static void __hoisted_1(uint32_t rows, uint32_t *gIn, uint32_t *gOut, uint32_t cols_sub2)
{
  uint32_t i = blockIdx.x / cols_sub2;
  uint32_t j = blockIdx.x % cols_sub2;
  gOut[j * (rows - (uint32_t)2U) + i] =
    gIn[i * (cols_sub2 + (uint32_t)2U) + j] +
      gIn[i * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)1U]
    + gIn[i * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)2U]
    + gIn[(i + (uint32_t)1U) * (cols_sub2 + (uint32_t)2U) + j]
    + gIn[(i + (uint32_t)1U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)1U] * 8U
    + gIn[(i + (uint32_t)1U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)2U]
    + gIn[(i + (uint32_t)2U) * (cols_sub2 + (uint32_t)2U) + j]
    + gIn[(i + (uint32_t)2U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)1U]
    + gIn[(i + (uint32_t)2U) * (cols_sub2 + (uint32_t)2U) + j + (uint32_t)2U];
}

void
Kuiper_Stencil_stencil3x3_i32_add_mul2_rc(
  uint32_t rows,
  uint32_t cols,
  uint32_t *gIn,
  uint32_t *gOut
)
{
  uint32_t cols_sub2 = cols - (uint32_t)2U;
  KPR_KCALL(__hoisted_1,
    (rows - (uint32_t)2U) * cols_sub2,
    (uint32_t)1U,
    (uint32_t)0U,
    rows,
    gIn,
    gOut,
    cols_sub2);
  cudaDeviceSynchronize();
}

