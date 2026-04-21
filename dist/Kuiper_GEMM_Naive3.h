
#ifndef Kuiper_GEMM_Naive3_H
#define Kuiper_GEMM_Naive3_H

#include <kuiper.h>

void
Kuiper_GEMM_Naive3_g_matmul_f32_rrr(uint32_t m,
                                    uint32_t n,
                                    uint32_t k,
                                    float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Naive3_g_matmul_f64_rrr(uint32_t m,
                                    uint32_t n,
                                    uint32_t k,
                                    double *gA, double *gB, double *gC);

#define Kuiper_GEMM_Naive3_H_DEFINED
#endif                          /* Kuiper_GEMM_Naive3_H */
