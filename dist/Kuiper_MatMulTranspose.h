
#ifndef Kuiper_MatMulTranspose_H
#define Kuiper_MatMulTranspose_H

#include <kuiper.h>

/**
An example of computing tr(AB) by just shifting a view. Basically:
  - Instantiating rA=rB=row_major, rC=col_major
  - Do the product, we get C = AB (in col-major)
  - View-shift C to get tr(AB) in row-major

TODO: It would be nicer to do this just over a CPU-side matmul, but there is no
view-like interface for CPU arrays.
*/
void
Kuiper_MatMulTranspose_matmul_transpose_gpu_f32_ff(uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   float *gA,
                                                   float *gB, float *gC);

#define Kuiper_MatMulTranspose_H_DEFINED
#endif                          /* Kuiper_MatMulTranspose_H */
