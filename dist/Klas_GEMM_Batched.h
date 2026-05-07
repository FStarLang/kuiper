
#ifndef Klas_GEMM_Batched_H
#define Klas_GEMM_Batched_H

#include <kuiper.h>

float
*Klas_GEMM_Batched_batched_gemm_f32(uint32_t batch,
                                    uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols, float *a, float *b);

#define Klas_GEMM_Batched_H_DEFINED
#endif                          /* Klas_GEMM_Batched_H */
