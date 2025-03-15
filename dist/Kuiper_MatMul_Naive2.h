

#ifndef __Kuiper_MatMul_Naive2_H
#define __Kuiper_MatMul_Naive2_H


#include <kuiper.h>

float_t
*Kuiper_MatMul_Naive2_matmul_f32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_MatMul_Naive2_matmul_f64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_MatMul_Naive2_matmul_u32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_MatMul_Naive2_matmul_u64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

float_t
*Kuiper_MatMul_Naive2_matmul_f32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
);

double_t
*Kuiper_MatMul_Naive2_matmul_f64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
);

uint32_t
*Kuiper_MatMul_Naive2_matmul_u32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
);

uint64_t
*Kuiper_MatMul_Naive2_matmul_u64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);


#define __Kuiper_MatMul_Naive2_H_DEFINED
#endif
