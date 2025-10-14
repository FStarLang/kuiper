
#ifndef Kuiper_MatMulTranspose_H
#define Kuiper_MatMulTranspose_H

#include <kuiper.h>

void
Kuiper_MatMulTranspose_matmul_transpose_gpu_f32_ff(uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols,
                                                   float_t * gA,
                                                   float_t * gB, float_t * gC);

#define Kuiper_MatMulTranspose_H_DEFINED
#endif                          /* Kuiper_MatMulTranspose_H */
