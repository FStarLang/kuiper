
#ifndef Klas_SPMM_H
#define Klas_SPMM_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_Matrix_smatrix__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__uint32_t;

typedef struct Kuiper_Sparse_Matrix_smatrix__float_s {
    uint32_t nnz;
    float *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__float;

#define Klas_SPMM_H_DEFINED
#endif                          /* Klas_SPMM_H */
