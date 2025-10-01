

#ifndef Kuiper_GEMM_Tiled_H
#define Kuiper_GEMM_Tiled_H

#include <kuiper.h>

float_t
*Kuiper_GEMM_Tiled_matmul_f32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_Tiled_matmul_f64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_Tiled_matmul_f32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_Tiled_matmul_f64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
);

void
Kuiper_GEMM_Tiled_g_matmul_f32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
);

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
);


#define Kuiper_GEMM_Tiled_H_DEFINED
#endif /* Kuiper_GEMM_Tiled_H */
