
#ifndef Kuiper_Sparse_SPMM_H
#define Kuiper_Sparse_SPMM_H

#include <kuiper.h>

typedef void *Kuiper_Sparse_SPMM_lseq;

typedef void *Kuiper_Sparse_SPMM_well_formed;

typedef struct Kuiper_Sparse_smatrix__uint32_t_s {
    uint32_t nnz1;
    uint32_t *elems1;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_smatrix__uint32_t;

void
Kuiper_Sparse_SPMM__spmm_u32(uint32_t rows,
                             uint32_t shared,
                             uint32_t cols,
                             Kuiper_Sparse_smatrix__uint32_t gA,
                             uint32_t * gB,
                             uint32_t * gC, uint32_t blockItemsK);

#define Kuiper_Sparse_SPMM_H_DEFINED
#endif                          /* Kuiper_Sparse_SPMM_H */
