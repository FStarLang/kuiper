
#ifndef Klas_GEMM_OrigBlockTiling1D_H
#define Klas_GEMM_OrigBlockTiling1D_H

#include <kuiper.h>

void
Klas_GEMM_OrigBlockTiling1D_matmul_f32_tiles64x8_8x64_rc8_rrr(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              float *gA,
                                                              float *gB,
                                                              float *gC);

void
Klas_GEMM_OrigBlockTiling1D_g_gemm_f32_tiles64x8_8x64_rc8_rrr(float alpha,
                                                              float beta,
                                                              uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              float *gA,
                                                              float *gB,
                                                              float *gC);

#define Klas_GEMM_OrigBlockTiling1D_H_DEFINED
#endif                          /* Klas_GEMM_OrigBlockTiling1D_H */
