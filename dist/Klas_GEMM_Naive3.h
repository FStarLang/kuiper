
#ifndef Klas_GEMM_Naive3_H
#define Klas_GEMM_Naive3_H

#include <kuiper.h>

void
Klas_GEMM_Naive3_g_matmul_f32_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC);

void
Klas_GEMM_Naive3_g_matmul_f64_rrr(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC);

void
Klas_GEMM_Naive3_g_matmul_f32_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k, float *gA, float *gB, float *gC);

void
Klas_GEMM_Naive3_g_matmul_f64_ccc(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  double *gA, double *gB, double *gC);

#define Klas_GEMM_Naive3_H_DEFINED
#endif                          /* Klas_GEMM_Naive3_H */
