
#ifndef Kuiper_GEMM_BlockTiling2D_H
#define Kuiper_GEMM_BlockTiling2D_H

#include <kuiper.h>

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x16(float alpha,
                                                      float beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float *gA,
                                                      float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x8(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x16(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x8(float alpha,
                                                     float beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     float *gA,
                                                     float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x16(float alpha,
                                                      float beta,
                                                      uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      float *gA,
                                                      float *gB, float *gC);

#define Kuiper_GEMM_BlockTiling2D_H_DEFINED
#endif                          /* Kuiper_GEMM_BlockTiling2D_H */
