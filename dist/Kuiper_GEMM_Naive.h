
#ifndef Kuiper_GEMM_Naive_H
#define Kuiper_GEMM_Naive_H

#include <kuiper.h>

void
Kuiper_GEMM_Naive_g_matmul_f32_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k, float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Naive_g_matmul_f64_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Naive_g_matmul_u32_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   uint32_t * gA, uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Naive_g_matmul_u64_rrr(uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   uint64_t * gA, uint64_t * gB, uint64_t * gC);

#define Kuiper_GEMM_Naive_H_DEFINED
#endif                          /* Kuiper_GEMM_Naive_H */
