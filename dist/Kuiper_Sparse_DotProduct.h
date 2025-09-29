

#ifndef Kuiper_Sparse_DotProduct_H
#define Kuiper_Sparse_DotProduct_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_sarray__uint32_t_s
{
  size_t nnz;
  size_t len;
  uint32_t *elems;
  size_t *pos;
}
Kuiper_Sparse_sarray__uint32_t;

uint32_t
Kuiper_Sparse_DotProduct_product_dense_u32(Kuiper_Sparse_sarray__uint32_t a, uint32_t *v);

uint32_t
Kuiper_Sparse_DotProduct_product_sparse_quadratic_u32(
  Kuiper_Sparse_sarray__uint32_t a,
  Kuiper_Sparse_sarray__uint32_t b
);

uint32_t
Kuiper_Sparse_DotProduct_product_sparse_u32(
  Kuiper_Sparse_sarray__uint32_t a,
  Kuiper_Sparse_sarray__uint32_t b
);


#define Kuiper_Sparse_DotProduct_H_DEFINED
#endif /* Kuiper_Sparse_DotProduct_H */
