
#ifndef Kuiper_GEMM_Naive_H
#define Kuiper_GEMM_Naive_H

#include <kuiper.h>

float
*Kuiper_GEMM_Naive_matmul_f32_rrr(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, float *a, float *b);

double
*Kuiper_GEMM_Naive_matmul_f64_rrr(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, double *a, double *b);

uint32_t
    * Kuiper_GEMM_Naive_matmul_u32_rrr(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols,
                                       uint32_t * a, uint32_t * b);

uint64_t
    * Kuiper_GEMM_Naive_matmul_u64_rrr(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols,
                                       uint64_t * a, uint64_t * b);

float
*Kuiper_GEMM_Naive_matmul_f32_ccc(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, float *a, float *b);

double
*Kuiper_GEMM_Naive_matmul_f64_ccc(uint32_t rows,
                                  uint32_t shared,
                                  uint32_t cols, double *a, double *b);

uint32_t
    * Kuiper_GEMM_Naive_matmul_u32_ccc(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols,
                                       uint32_t * a, uint32_t * b);

uint64_t
    * Kuiper_GEMM_Naive_matmul_u64_ccc(uint32_t rows,
                                       uint32_t shared,
                                       uint32_t cols,
                                       uint64_t * a, uint64_t * b);

void
Kuiper_GEMM_Naive_g_matmul_f32_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Naive_g_matmul_f64_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Naive_g_matmul_u32_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint32_t * gA, uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Naive_g_matmul_u64_rrr(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint64_t * gA, uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_Naive_g_matmul_f32_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Naive_g_matmul_f64_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Naive_g_matmul_u32_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint32_t * gA, uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Naive_g_matmul_u64_ccc(uint32_t rows,
                                   uint32_t shared,
                                   uint32_t cols,
                                   uint64_t * gA, uint64_t * gB, uint64_t * gC);

#define Kuiper_GEMM_Naive_H_DEFINED
#endif                          /* Kuiper_GEMM_Naive_H */
