

#include "Kuiper_Poly_Stencil.h"

__global__
/**
  hoisted when extracting stencil_add_only
*/
static void __hoisted_0(size_t cols, float_t *gIn, float_t *gOut)
{
  size_t i = blockIdx.x / cols;
  size_t j = blockIdx.x % cols;
  gOut[i * cols + j] =
    gIn[i * (cols + (size_t)2U) + j] * (float_t)1.0f +
      gIn[i * (cols + (size_t)2U) + j + (size_t)1U] * (float_t)1.0f
    + gIn[i * (cols + (size_t)2U) + j + (size_t)2U] * (float_t)1.0f
    + gIn[(i + (size_t)1U) * (cols + (size_t)2U) + j] * (float_t)1.0f
    + gIn[(i + (size_t)1U) * (cols + (size_t)2U) + j + (size_t)1U] * (float_t)1.0f
    + gIn[(i + (size_t)1U) * (cols + (size_t)2U) + j + (size_t)2U] * (float_t)1.0f
    + gIn[(i + (size_t)2U) * (cols + (size_t)2U) + j] * (float_t)1.0f
    + gIn[(i + (size_t)2U) * (cols + (size_t)2U) + j + (size_t)1U] * (float_t)1.0f
    + gIn[(i + (size_t)2U) * (cols + (size_t)2U) + j + (size_t)2U] * (float_t)1.0f;
}

void
Kuiper_Poly_Stencil_stencil_add_only(size_t rows, size_t cols, float_t *gIn, float_t *gOut)
{
  KPR_KCALL(__hoisted_0, rows * cols, (size_t)1U, (size_t)1U, (size_t)0U, cols, gIn, gOut);
  cudaDeviceSynchronize();
}

