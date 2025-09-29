

#ifndef Kuiper_GEMM_TensorCore_H
#define Kuiper_GEMM_TensorCore_H

#include <kuiper.h>

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x16_16x16x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_32x8x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_8x32x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x8x16_32x8x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_8x32x16_8x32x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_32x8x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_8x32x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_16x16x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f16_16x16x16_16x16x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  half_t *gC
);

void
Kuiper_GEMM_TensorCore_g_gemm_f16_f32_32x32x32_16x16x16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  half_t *gA,
  half_t *gB,
  float_t *gC
);


#define Kuiper_GEMM_TensorCore_H_DEFINED
#endif /* Kuiper_GEMM_TensorCore_H */
