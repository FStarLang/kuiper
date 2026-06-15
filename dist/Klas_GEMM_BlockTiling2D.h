
#ifndef Klas_GEMM_BlockTiling2D_H
#define Klas_GEMM_BlockTiling2D_H

#include <kuiper.h>

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x32x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x64x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x64x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_32x128x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x128x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x32x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x32x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x32_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x32_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x8(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_8x8(__nv_bfloat16 alpha,
                                                 __nv_bfloat16 beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 __nv_bfloat16 * gA,
                                                 __nv_bfloat16 * gB,
                                                 __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_8x16(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_8x16(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_16x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x64x64_16x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x64x64_16x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_64x128x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_64x128x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x32x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x32x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x32_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x32_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x8(float alpha,
                                                 float beta,
                                                 uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *gA,
                                                 float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_8x8(__nv_bfloat16 alpha,
                                                  __nv_bfloat16 beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  __nv_bfloat16 * gA,
                                                  __nv_bfloat16 * gB,
                                                  __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_8x16(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_8x16(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_16x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x64x64_16x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x64x64_16x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_8x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_8x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_16x8(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x32_16x16(__nv_bfloat16 alpha,
                                                     __nv_bfloat16 beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     __nv_bfloat16 * gA,
                                                     __nv_bfloat16 * gB,
                                                     __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x8(float alpha,
                                                  float beta,
                                                  uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_8x8(__nv_bfloat16 alpha,
                                                   __nv_bfloat16 beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 * gA,
                                                   __nv_bfloat16 * gB,
                                                   __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_8x16(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_8x16(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x8(float alpha,
                                                   float beta,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float *gA,
                                                   float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_16x8(__nv_bfloat16 alpha,
                                                    __nv_bfloat16 beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    __nv_bfloat16 * gA,
                                                    __nv_bfloat16 * gB,
                                                    __nv_bfloat16 * gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_f32_128x128x64_16x16(float alpha,
                                                    float beta,
                                                    uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    float *gA,
                                                    float *gB, float *gC);

void
Klas_GEMM_BlockTiling2D_g_gemm_bf16_128x128x64_16x16(__nv_bfloat16 alpha,
                                                     __nv_bfloat16 beta,
                                                     uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     __nv_bfloat16 * gA,
                                                     __nv_bfloat16 * gB,
                                                     __nv_bfloat16 * gC);

#define Klas_GEMM_BlockTiling2D_H_DEFINED
#endif                          /* Klas_GEMM_BlockTiling2D_H */
