
#ifndef Kuiper_GEMM_Tiled_H
#define Kuiper_GEMM_Tiled_H

#include <kuiper.h>

void
Kuiper_GEMM_Tiled_g_matmul_f32_rrr(uint32_t tile,
                                   uint32_t m,
                                   uint32_t n,
                                   uint32_t k, float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Tiled_g_matmul_f64_rrr(uint32_t tile,
                                   uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Tiled_g_matmul_u32_rrr(uint32_t tile,
                                   uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   uint32_t * gA, uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Tiled_g_matmul_u64_rrr(uint32_t tile,
                                   uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   uint64_t * gA, uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          uint32_t * gA,
                                          uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          uint64_t * gA,
                                          uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          uint32_t * gA,
                                          uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_rrr(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          uint64_t * gA,
                                          uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_Tiled_g_gemm_f32_rrr(uint32_t tile,
                                 float alpha,
                                 float beta,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k, float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Tiled_g_gemm_f64_rrr(uint32_t tile,
                                 double alpha,
                                 double beta,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Tiled_g_gemm_u32_rrr(uint32_t tile,
                                 uint32_t alpha,
                                 uint32_t beta,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint32_t * gA, uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Tiled_g_gemm_u64_rrr(uint32_t tile,
                                 uint64_t alpha,
                                 uint64_t beta,
                                 uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 uint64_t * gA, uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_Tiled_g_gemm_f32_tile32_rrr(float alpha,
                                        float beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Tiled_g_gemm_f64_tile32_rrr(double alpha,
                                        double beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Tiled_g_gemm_u32_tile32_rrr(uint32_t alpha,
                                        uint32_t beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t * gA,
                                        uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Tiled_g_gemm_u64_tile32_rrr(uint64_t alpha,
                                        uint64_t beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint64_t * gA,
                                        uint64_t * gB, uint64_t * gC);

void
Kuiper_GEMM_Tiled_g_gemm_f32_tile16_rrr(float alpha,
                                        float beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        float *gA, float *gB, float *gC);

void
Kuiper_GEMM_Tiled_g_gemm_f64_tile16_rrr(double alpha,
                                        double beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        double *gA, double *gB, double *gC);

void
Kuiper_GEMM_Tiled_g_gemm_u32_tile16_rrr(uint32_t alpha,
                                        uint32_t beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t * gA,
                                        uint32_t * gB, uint32_t * gC);

void
Kuiper_GEMM_Tiled_g_gemm_u64_tile16_rrr(uint64_t alpha,
                                        uint64_t beta,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint64_t * gA,
                                        uint64_t * gB, uint64_t * gC);

#define Kuiper_GEMM_Tiled_H_DEFINED
#endif                          /* Kuiper_GEMM_Tiled_H */
