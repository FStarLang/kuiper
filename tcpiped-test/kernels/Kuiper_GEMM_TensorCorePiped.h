
#ifndef Kuiper_GEMM_TensorCorePiped_H
#define Kuiper_GEMM_TensorCorePiped_H

#include <kuiper.h>

void
Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t * gA,
                                                              half_t * gB,
                                                              half_t * gC);

void
Kuiper_GEMM_TensorCorePiped_change_sync_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC);

void
Kuiper_GEMM_TensorCorePiped_align_g_gemm_f16_f16_64x64x32_16x16x16_4x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half_t *gA,
                                                              half_t *gB,
                                                              half_t *gC);

void
Kuiper_GEMM_TensorCorePiped_align_g_gemm_f16_f16_128x128x64_16x16x16_4x8(uint32_t rows,
                                                                uint32_t shared,
                                                                uint32_t cols,
                                                                half_t *gA,
                                                                half_t *gB,
                                                                half_t *gC);

#define Kuiper_GEMM_TensorCorePiped_H_DEFINED
#endif
