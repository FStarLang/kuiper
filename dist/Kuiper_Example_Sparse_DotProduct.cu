
#include "Kuiper_Example_Sparse_DotProduct.h"

uint32_t
Kuiper_Example_Sparse_DotProduct_sarray_product_dense_u32
(Kuiper_Sparse_Array_sarray__uint32_t a, uint32_t *v)
{
    uint32_t i = 0U;
    uint32_t dp = 0U;
    for (; i < a.nnz; i++)
        dp += a.elems[i] * v[a.pos[i]];
    return dp;
}
