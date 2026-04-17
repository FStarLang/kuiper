
#ifndef Kuiper_Sparse_MM_H
#define Kuiper_Sparse_MM_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_Matrix_smatrix__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__uint32_t;

void
Kuiper_Sparse_MM__mmsd_u32_rr(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                              uint32_t * gB, uint32_t * gC);

void
Kuiper_Sparse_MM__mmsd_u32_cc(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                              uint32_t * gB, uint32_t * gC);

#define Kuiper_Sparse_MM_H_DEFINED
#endif                          /* Kuiper_Sparse_MM_H */
