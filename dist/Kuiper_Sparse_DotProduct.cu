

#include "Kuiper_Sparse_DotProduct.h"

uint32_t
Kuiper_Sparse_DotProduct_product_dense_u32(Kuiper_Sparse_sarray__uint32_t a, uint32_t *v)
{
  size_t i = (size_t)0U;
  uint32_t dp = 0U;
  for (; i < a.nnz; i += (size_t)1U)
    dp += a.elems[i] * v[a.pos[i]];
  return dp;
}

uint32_t
Kuiper_Sparse_DotProduct_product_sparse_quadratic_u32(
  Kuiper_Sparse_sarray__uint32_t a,
  Kuiper_Sparse_sarray__uint32_t b
)
{
  uint32_t dp = 0U;
  size_t i = (size_t)0U;
  while (i < a.nnz)
  {
    size_t j = (size_t)0U;
    size_t p_a = a.pos[i];
    while (j < b.nnz)
      if (p_a == b.pos[j])
      {
        dp += a.elems[i] * b.elems[j];
        j += (size_t)1U;
      }
      else
        j += (size_t)1U;
    i += (size_t)1U;
  }
  return dp;
}

uint32_t
Kuiper_Sparse_DotProduct_product_sparse_u32(
  Kuiper_Sparse_sarray__uint32_t a,
  Kuiper_Sparse_sarray__uint32_t b
)
{
  uint32_t dp = 0U;
  size_t i = (size_t)0U;
  size_t j = (size_t)0U;
  while (i < a.nnz && j < b.nnz)
  {
    size_t p_a = a.pos[i];
    size_t p_b = b.pos[j];
    if (p_a < p_b)
      i += (size_t)1U;
    else if (p_b < p_a)
      j += (size_t)1U;
    else
    {
      dp += a.elems[i] * b.elems[j];
      i += (size_t)1U;
      j += (size_t)1U;
    }
  }
  return dp;
}

