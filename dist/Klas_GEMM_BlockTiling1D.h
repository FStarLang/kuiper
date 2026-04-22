
#ifndef Klas_GEMM_BlockTiling1D_H
#define Klas_GEMM_BlockTiling1D_H

#include <kuiper.h>

void
Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                double *gA,
                                                double *gB, double *gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t * gA,
                                                uint32_t * gB, uint32_t * gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint64_t * gA,
                                                uint64_t * gB, uint64_t * gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                float *gA,
                                                float *gB, float *gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                double *gA,
                                                double *gB, double *gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t * gA,
                                                uint32_t * gB, uint32_t * gC);

void
Klas_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint64_t * gA,
                                                uint64_t * gB, uint64_t * gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(float alpha,
                                              float beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              float *gA, float *gB, float *gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(double alpha,
                                              double beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              double *gA,
                                              double *gB, double *gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                              uint32_t beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              uint32_t * gA,
                                              uint32_t * gB, uint32_t * gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                              uint64_t beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              uint64_t * gA,
                                              uint64_t * gB, uint64_t * gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(float alpha,
                                              float beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              float *gA, float *gB, float *gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(double alpha,
                                              double beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              double *gA,
                                              double *gB, double *gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                              uint32_t beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              uint32_t * gA,
                                              uint32_t * gB, uint32_t * gC);

void
Klas_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                              uint64_t beta,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              uint64_t * gA,
                                              uint64_t * gB, uint64_t * gC);

#define Klas_GEMM_BlockTiling1D_H_DEFINED
#endif                          /* Klas_GEMM_BlockTiling1D_H */
