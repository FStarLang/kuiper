
#ifndef Kuiper_GEMM_BlockTiling1D_H
#define Kuiper_GEMM_BlockTiling1D_H

#include <kuiper.h>

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile32_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b);

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile32_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b);

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile32_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t * a,
                                                      uint32_t * b);

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile32_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t * a,
                                                      uint64_t * b);

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile32_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b);

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile32_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b);

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile32_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t * a,
                                                      uint32_t * b);

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile32_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t * a,
                                                      uint64_t * b);

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile16_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b);

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile16_rrr(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b);

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile16_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t * a,
                                                      uint32_t * b);

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile16_rrr(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t * a,
                                                      uint64_t * b);

float
*Kuiper_GEMM_BlockTiling1D_matmul_f32_tile16_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 float *a, float *b);

double
*Kuiper_GEMM_BlockTiling1D_matmul_f64_tile16_ccc(uint32_t rows,
                                                 uint32_t shared,
                                                 uint32_t cols,
                                                 double *a, double *b);

uint32_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u32_tile16_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint32_t * a,
                                                      uint32_t * b);

uint64_t
    * Kuiper_GEMM_BlockTiling1D_matmul_u64_tile16_ccc(uint32_t rows,
                                                      uint32_t shared,
                                                      uint32_t cols,
                                                      uint64_t * a,
                                                      uint64_t * b);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t * gA,
                                                  uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t * gA,
                                                  uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t * gA,
                                                  uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile32_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t * gA,
                                                  uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t * gA,
                                                  uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_rrr(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t * gA,
                                                  uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f32_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  float *gA,
                                                  float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_f64_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  double *gA,
                                                  double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u32_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint32_t * gA,
                                                  uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_matmul_u64_tile16_ccc(uint32_t rows,
                                                  uint32_t shared,
                                                  uint32_t cols,
                                                  uint64_t * gA,
                                                  uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_rrr(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_rrr(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t * gA,
                                                uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t * gA,
                                                uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile32_ccc(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile32_ccc(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile32_ccc(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t * gA,
                                                uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile32_ccc(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t * gA,
                                                uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_rrr(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_rrr(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t * gA,
                                                uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t * gA,
                                                uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f32_tile16_ccc(float alpha,
                                                float beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                float *gA,
                                                float *gB, float *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_f64_tile16_ccc(double alpha,
                                                double beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                double *gA,
                                                double *gB, double *gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u32_tile16_ccc(uint32_t alpha,
                                                uint32_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint32_t * gA,
                                                uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_BlockTiling1D_g_gemm_u64_tile16_ccc(uint64_t alpha,
                                                uint64_t beta,
                                                uint32_t rows,
                                                uint32_t shared,
                                                uint32_t cols,
                                                uint64_t * gA,
                                                uint64_t * gB, uint64_t * gC);

#define Kuiper_GEMM_BlockTiling1D_H_DEFINED
#endif                          /* Kuiper_GEMM_BlockTiling1D_H */
