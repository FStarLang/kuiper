
#ifndef Klas_GEMM_TensorCore_H
#define Klas_GEMM_TensorCore_H

#include <kuiper.h>

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x16_16x16x16(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      half * gA,
                                                      half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_32x8x16(uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half * gA,
                                                     half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_8x32x16(uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half * gA,
                                                     half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_32x8x16_32x8x16(uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    half * gA,
                                                    half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_8x32x16_8x32x16(uint32_t rows,
                                                    uint32_t shared,
                                                    uint32_t cols,
                                                    half * gA,
                                                    half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      half * gA,
                                                      half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_32x8x16(uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half * gA,
                                                     half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_8x32x16(uint32_t rows,
                                                     uint32_t shared,
                                                     uint32_t cols,
                                                     half * gA,
                                                     half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_32x32x32_16x16x16(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      half * gA,
                                                      half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f16_16x16x16_16x16x16(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      half * gA,
                                                      half * gB, half * gC);

void
Klas_GEMM_TensorCore_g_gemm_f16_f32_32x32x32_16x16x16(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      half * gA,
                                                      half * gB, float *gC);

#define Klas_GEMM_TensorCore_H_DEFINED
#endif                          /* Klas_GEMM_TensorCore_H */
