

#ifndef __Kuiper_GEMM_BlockTiling2D_H
#define __Kuiper_GEMM_BlockTiling2D_H

#include <kuiper.h>

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_64x64x8_8x8_rrr_rr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

float_t
*Kuiper_GEMM_BlockTiling2D_matmul_f32_32x32x32_32x8_rrr_rr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rrr_rr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr_rr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rrr_cr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);


#define __Kuiper_GEMM_BlockTiling2D_H_DEFINED
#endif
