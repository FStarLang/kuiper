
#ifndef Kuiper_GEMM_TensorCore2D_H
#define Kuiper_GEMM_TensorCore2D_H

#include <kuiper.h>

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_2x2(uint32_t bm,
                                                     uint32_t bn,
                                                     uint32_t bk,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half_t * gA,
                                                     half_t * gB, half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_4x4(uint32_t bm,
                                                     uint32_t bn,
                                                     uint32_t bk,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half_t * gA,
                                                     half_t * gB, half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_16x16x16_8x8(uint32_t bm,
                                                     uint32_t bn,
                                                     uint32_t bk,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half_t * gA,
                                                     half_t * gB, half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x16_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x32x64_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x64x16_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x64x16_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x64x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x64x32_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x64x64_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x64x64_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x16_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x16_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x32_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x32_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x32_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x64_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x64_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_32x128x64_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x32x16_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x32x16_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x32x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x32x32_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x32x64_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x32x64_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x16_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_2x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x2(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x64_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x16_16x16x16_4x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x32_16x16x16_4x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_2x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x128x64_16x16x16_4x8(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x16_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x16_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x32_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x32_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x32_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x64_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x64_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x32x64_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x16_16x16x16_8x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x32_16x16x16_8x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_2x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_2x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_4x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_4x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_8x2(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x64x64_16x16x16_8x4(uint32_t rows,
                                                               uint32_t shared,
                                                               uint32_t cols,
                                                               half_t * gA,
                                                               half_t * gB,
                                                               half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_2x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_2x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_16x16x16_8x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_2x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_2x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x2(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x4(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x64_16x16x16_8x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t * gA,
                                                                half_t * gB,
                                                                half_t * gC);

#define Kuiper_GEMM_TensorCore2D_H_DEFINED
#endif                          /* Kuiper_GEMM_TensorCore2D_H */
