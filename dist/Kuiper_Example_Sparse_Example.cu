
#include "Kuiper_Example_Sparse_Example.h"

void Kuiper_Example_Sparse_Example__id_u32(Kuiper_Sparse_Array_sarray__uint32_t
                                           a)
{
    uint32_t i = 0U;
    for (; i < a.nnz; i++)
        a.elems[i] = a.elems[i];
}

void
Kuiper_Example_Sparse_Example__scale_u32(uint32_t k,
                                         Kuiper_Sparse_Array_sarray__uint32_t a)
{
    uint32_t i = 0U;
    for (; i < a.nnz; i++)
        a.elems[i] *= k;
}
