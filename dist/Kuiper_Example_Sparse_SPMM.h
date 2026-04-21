
#ifndef Kuiper_Example_Sparse_SPMM_H
#define Kuiper_Example_Sparse_SPMM_H

#include <kuiper.h>

typedef struct Kuiper_Example_Sparse_SPMM_parameters_s {
    uint32_t rows;
    uint32_t shared;
    uint32_t cols;
    uint32_t blockItemsK;
    uint32_t blockItemsX;
    uint32_t blockWidth;
} Kuiper_Example_Sparse_SPMM_parameters;

typedef void *Kuiper_Example_Sparse_SPMM_lseq;

typedef void *Kuiper_Example_Sparse_SPMM_well_formed;

typedef struct Kuiper_Sparse_Matrix_smatrix__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__uint32_t;

void
Kuiper_Example_Sparse_SPMM_spmm_u32(uint32_t rows,
                                    uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                                    uint32_t * gB, uint32_t * gC);

#define Kuiper_Example_Sparse_SPMM_H_DEFINED
#endif                          /* Kuiper_Example_Sparse_SPMM_H */
