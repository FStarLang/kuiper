

#ifndef Kuiper_GEMM_BlockTiling2D_H
#define Kuiper_GEMM_BlockTiling2D_H

#include <kuiper.h>

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_rr(
  float_t alpha,
  float_t beta,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_rr(
  float_t alpha,
  float_t beta,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_cr(
  float_t alpha,
  float_t beta,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);


#define Kuiper_GEMM_BlockTiling2D_H_DEFINED
#endif /* Kuiper_GEMM_BlockTiling2D_H */
