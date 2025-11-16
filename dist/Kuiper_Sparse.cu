
#include "Kuiper_Sparse.h"

uint32_t Kuiper_Sparse_uu___0 = 1U;

void Kuiper_Sparse_smatrix_id_u32(Kuiper_Sparse_smatrix__uint32_t m)
{
    uint32_t i = 0U;
    for (; i < m.nnz1; i++)
        m.elems1[i] = m.elems1[i];
}

void Kuiper_Sparse_sarray_iterator_test_u32(Kuiper_Sparse_sarray__uint32_t a)
{
    uint32_t it = 0U;
    for (; !(it == a.nnz); it++) {

    }
}
