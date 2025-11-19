
#ifndef Kuiper_GEMM_OrigBlockTiling1D_H
#define Kuiper_GEMM_OrigBlockTiling1D_H

#include <kuiper.h>

void
Kuiper_GEMM_OrigBlockTiling1D_matmul_f32_tiles64x8_8x64_rc8_rrr(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                float *gA,
                                                                float *gB,
                                                                float *gC);

void
Kuiper_GEMM_OrigBlockTiling1D_g_gemm_f32_tiles64x8_8x64_rc8_rrr(float alpha,
                                                                float beta,
                                                                uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                float *gA,
                                                                float *gB,
                                                                float *gC);

#define Kuiper_GEMM_OrigBlockTiling1D_H_DEFINED
#endif                          /* Kuiper_GEMM_OrigBlockTiling1D_H */
