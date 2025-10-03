
#ifndef Kuiper_GEMM_BlockTiling2D_H
#define Kuiper_GEMM_BlockTiling2D_H

#include <kuiper.h>

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_8x8_cr(uint32_t bm,
                                            uint32_t bn,
                                            uint32_t bk,
                                            float_t alpha,
                                            float_t beta,
                                            uint32_t rows,
                                            uint32_t shared,
                                            uint32_t cols,
                                            float_t * gA,
                                            float_t * gB, float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x8_8x8_cr(float_t alpha,
                                                    float_t beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float_t * gA,
                                                    float_t * gB, float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x16_8x8_cr(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t * gA,
                                                     float_t * gB,
                                                     float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8_cr(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t * gA,
                                                     float_t * gB,
                                                     float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8_cr(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t * gA,
                                                     float_t * gB,
                                                     float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x8_8x8_cr(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t * gA,
                                                     float_t * gB,
                                                     float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x16_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x8_8x8_cr(float_t alpha,
                                                     float_t beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float_t * gA,
                                                     float_t * gB,
                                                     float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x16_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x8_8x8_cr(float_t alpha,
                                                      float_t beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float_t * gA,
                                                      float_t * gB,
                                                      float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x16_8x8_cr(float_t alpha,
                                                       float_t beta,
                                                       uint32_t rows,
                                                       uint32_t shared,
                                                       uint32_t cols,
                                                       float_t * gA,
                                                       float_t * gB,
                                                       float_t * gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8_cr(float_t alpha,
                                                       float_t beta,
                                                       uint32_t rows,
                                                       uint32_t shared,
                                                       uint32_t cols,
                                                       float_t * gA,
                                                       float_t * gB,
                                                       float_t * gC);

#define Kuiper_GEMM_BlockTiling2D_H_DEFINED
#endif                          /* Kuiper_GEMM_BlockTiling2D_H */
