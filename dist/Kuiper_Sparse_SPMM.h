
#ifndef Kuiper_Sparse_SPMM_H
#define Kuiper_Sparse_SPMM_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_Matrix_smatrix__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__uint32_t;

void
Kuiper_Sparse_SPMM_spmm_u32(uint32_t rows,
                            uint32_t shared,
                            uint32_t cols,
                            Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                            uint32_t * row_indices,
                            uint32_t * gB, uint32_t * gC);

typedef struct Kuiper_Sparse_Matrix_smatrix__float_s {
    uint32_t nnz;
    float *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__float;

void
Kuiper_Sparse_SPMM_spmm_f32(uint32_t rows,
                            uint32_t shared,
                            uint32_t cols,
                            Kuiper_Sparse_Matrix_smatrix__float gA,
                            uint32_t * row_indices, float *gB, float *gC);

#define Kuiper_Sparse_SPMM_H_DEFINED
#endif                          /* Kuiper_Sparse_SPMM_H */
