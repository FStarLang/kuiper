

#ifndef __Kuiper_MatMul_Tiled_H
#define __Kuiper_MatMul_Tiled_H


#include <kuiper.h>

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_rrr(
  size_t tile,
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);

uint64_t
*Kuiper_MatMul_Tiled_matmul_u64_rrr_tile32(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
);


#define __Kuiper_MatMul_Tiled_H_DEFINED
#endif
