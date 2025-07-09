

#include "Kuiper_Stencil.h"

__global__
/**
  hoisted when extracting stencil3x3_f32_add_rr
*/
static void __hoisted_0(float_t *gIn, float_t *gOut, size_t cols_sub2)
{
  size_t i = blockIdx.x / cols_sub2;
  size_t j = blockIdx.x % cols_sub2;
  gOut[i * cols_sub2 + j] =
    gIn[i * (cols_sub2 + (size_t)2U) + j] * (float_t)1.0f +
      gIn[i * (cols_sub2 + (size_t)2U) + j + (size_t)1U] * (float_t)1.0f
    + gIn[i * (cols_sub2 + (size_t)2U) + j + (size_t)2U] * (float_t)1.0f
    + gIn[(i + (size_t)1U) * (cols_sub2 + (size_t)2U) + j] * (float_t)1.0f
    + gIn[(i + (size_t)1U) * (cols_sub2 + (size_t)2U) + j + (size_t)1U] * (float_t)1.0f
    + gIn[(i + (size_t)1U) * (cols_sub2 + (size_t)2U) + j + (size_t)2U] * (float_t)1.0f
    + gIn[(i + (size_t)2U) * (cols_sub2 + (size_t)2U) + j] * (float_t)1.0f
    + gIn[(i + (size_t)2U) * (cols_sub2 + (size_t)2U) + j + (size_t)1U] * (float_t)1.0f
    + gIn[(i + (size_t)2U) * (cols_sub2 + (size_t)2U) + j + (size_t)2U] * (float_t)1.0f;
}

void
Kuiper_Stencil_stencil3x3_f32_add_rr(size_t rows, size_t cols, float_t *gIn, float_t *gOut)
{
  size_t cols_sub2 = cols - (size_t)2U;
  KPR_KCALL(__hoisted_0,
    (rows - (size_t)2U) * cols_sub2,
    (size_t)1U,
    (size_t)1U,
    (size_t)0U,
    gIn,
    gOut,
    cols_sub2);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting stencil3x3_i32_add_mul2_rc
*/
static void __hoisted_1(size_t rows, uint32_t *gIn, uint32_t *gOut, size_t cols_sub2)
{
  size_t i = blockIdx.x / cols_sub2;
  size_t j = blockIdx.x % cols_sub2;
  gOut[j * (rows - (size_t)2U) + i] =
    gIn[i * (cols_sub2 + (size_t)2U) + j] + gIn[i * (cols_sub2 + (size_t)2U) + j + (size_t)1U] +
      gIn[i * (cols_sub2 + (size_t)2U) + j + (size_t)2U]
    + gIn[(i + (size_t)1U) * (cols_sub2 + (size_t)2U) + j]
    + gIn[(i + (size_t)1U) * (cols_sub2 + (size_t)2U) + j + (size_t)1U] * 8U
    + gIn[(i + (size_t)1U) * (cols_sub2 + (size_t)2U) + j + (size_t)2U]
    + gIn[(i + (size_t)2U) * (cols_sub2 + (size_t)2U) + j]
    + gIn[(i + (size_t)2U) * (cols_sub2 + (size_t)2U) + j + (size_t)1U]
    + gIn[(i + (size_t)2U) * (cols_sub2 + (size_t)2U) + j + (size_t)2U];
}

void
Kuiper_Stencil_stencil3x3_i32_add_mul2_rc(
  size_t rows,
  size_t cols,
  uint32_t *gIn,
  uint32_t *gOut
)
{
  size_t cols_sub2 = cols - (size_t)2U;
  KPR_KCALL(__hoisted_1,
    (rows - (size_t)2U) * cols_sub2,
    (size_t)1U,
    (size_t)1U,
    (size_t)0U,
    rows,
    gIn,
    gOut,
    cols_sub2);
  cudaDeviceSynchronize();
}

