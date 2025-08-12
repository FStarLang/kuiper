

#ifndef Kuiper_GEMM_SHMem_H
#define Kuiper_GEMM_SHMem_H

#include <kuiper.h>

float_t
*Kuiper_GEMM_SHMem_matmul_f32_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_SHMem_matmul_f64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_SHMem_matmul_f32_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_SHMem_matmul_f64_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_SHMem_matmul_f32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_SHMem_matmul_f64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_SHMem_matmul_u32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_SHMem_matmul_u64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

void
Kuiper_GEMM_SHMem_g_matmul_f32_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u32_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f32_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f64_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u32_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u64_ccc(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile16_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_f64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u32_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_matmul_u64_tile16_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f32_rrr(
  size_t tile,
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f64_rrr(
  size_t tile,
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u32_rrr(
  size_t tile,
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u64_rrr(
  size_t tile,
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f32_ccc(
  size_t tile,
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f64_ccc(
  size_t tile,
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u32_ccc(
  size_t tile,
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u64_ccc(
  size_t tile,
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile32_rrr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile32_rrr(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile32_rrr(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile32_rrr(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile32_ccc(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile32_ccc(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile32_ccc(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile32_ccc(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile16_rrr(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile16_rrr(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile16_rrr(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile16_rrr(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f32_tile16_ccc(
  float_t alpha,
  float_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_f64_tile16_ccc(
  double_t alpha,
  double_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u32_tile16_ccc(
  uint32_t alpha,
  uint32_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_SHMem_g_gemm_u64_tile16_ccc(
  uint64_t alpha,
  uint64_t beta,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);


#define Kuiper_GEMM_SHMem_H_DEFINED
#endif /* Kuiper_GEMM_SHMem_H */
