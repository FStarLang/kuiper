
#ifndef Klas_GEMM_Batched_H
#define Klas_GEMM_Batched_H

#include <kuiper.h>

void
Klas_GEMM_Batched_batched_gemm_f32(float alpha,
                                   float beta,
                                   uint32_t batch,
                                   uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols, float *a, float *b, float *c);

float
*Klas_GEMM_Batched_batched_matmul_f32(uint32_t batch,
                                      uint32_t rows,
                                      uint32_t shared,
                                      uint32_t cols, float *a, float *b);

#define Klas_GEMM_Batched_H_DEFINED
#endif                          /* Klas_GEMM_Batched_H */
