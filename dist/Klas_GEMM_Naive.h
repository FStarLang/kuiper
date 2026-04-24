
#ifndef Klas_GEMM_Naive_H
#define Klas_GEMM_Naive_H

#include <kuiper.h>

void
Klas_GEMM_Naive_g_matmul_f32_rrr(uint32_t m,
                                 uint32_t n,
                                 uint32_t k, float *gA, float *gB, float *gC);

void
Klas_GEMM_Naive_g_matmul_f64_rrr(uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 double *gA, double *gB, double *gC);

void
Klas_GEMM_Naive_g_matmul_u32_rrr(uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint32_t * gA, uint32_t * gB, uint32_t * gC);

void
Klas_GEMM_Naive_g_matmul_u64_rrr(uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint64_t * gA, uint64_t * gB, uint64_t * gC);

#define Klas_GEMM_Naive_H_DEFINED
#endif                          /* Klas_GEMM_Naive_H */
