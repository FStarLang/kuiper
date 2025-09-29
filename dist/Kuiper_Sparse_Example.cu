

#include "Kuiper_Sparse_Example.h"

void Kuiper_Sparse_Example__id_u32(Kuiper_Sparse_sarray__uint32_t a)
{
  size_t i = (size_t)0U;
  for (; i < a.nnz; i += (size_t)1U)
    a.elems[i] = a.elems[i];
}

void Kuiper_Sparse_Example__scale_u32(uint32_t k, Kuiper_Sparse_sarray__uint32_t a)
{
  size_t i = (size_t)0U;
  for (; i < a.nnz; i += (size_t)1U)
    a.elems[i] *= k;
}

