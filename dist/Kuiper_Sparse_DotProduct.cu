
#include "Kuiper_Sparse_DotProduct.h"

uint32_t
Kuiper_Sparse_DotProduct_product_dense_u32(Kuiper_Sparse_sarray__uint32_t a,
                                           uint32_t *v)
{
    uint32_t i = 0U;
    uint32_t dp = 0U;
    for (; i < a.nnz; i++)
        dp += a.elems[i] * v[a.pos[i]];
    return dp;
}

uint32_t
Kuiper_Sparse_DotProduct_product_sparse_quadratic_u32
(Kuiper_Sparse_sarray__uint32_t a, Kuiper_Sparse_sarray__uint32_t b)
{
    uint32_t dp = 0U;
    uint32_t i = 0U;
    for (; i < a.nnz; i++) {
        uint32_t j = 0U;
        uint32_t p_a = a.pos[i];
        while (j < b.nnz)
            if (p_a == b.pos[j]) {
                dp += a.elems[i] * b.elems[j];
                j++;
            } else
                j++;
    }
    return dp;
}

uint32_t
Kuiper_Sparse_DotProduct_product_sparse_u32(Kuiper_Sparse_sarray__uint32_t a,
                                            Kuiper_Sparse_sarray__uint32_t b)
{
    uint32_t dp = 0U;
    uint32_t i = 0U;
    uint32_t j = 0U;
    while (i < a.nnz && j < b.nnz) {
        uint32_t p_a = a.pos[i];
        uint32_t p_b = b.pos[j];
        if (p_a < p_b)
            i++;
        else if (p_b < p_a)
            j++;
        else {
            dp += a.elems[i] * b.elems[j];
            i++;
            j++;
        }
    }
    return dp;
}
