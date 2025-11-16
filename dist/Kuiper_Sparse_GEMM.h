
#ifndef Kuiper_Sparse_GEMM_H
#define Kuiper_Sparse_GEMM_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_smatrix__uint32_t_s {
    uint32_t nnz1;
    uint32_t *elems1;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_smatrix__uint32_t;

void
Kuiper_Sparse_GEMM__gemm_u32_rr(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_smatrix__uint32_t gA,
                                uint32_t * gB, uint32_t * gC);

#define Kuiper_Sparse_GEMM_H_DEFINED
#endif                          /* Kuiper_Sparse_GEMM_H */
