
#ifndef Kuiper_Sparse_H
#define Kuiper_Sparse_H

#include <kuiper.h>

extern uint32_t Kuiper_Sparse_x;

typedef uint32_t Kuiper_Sparse_sarray_iterator;

typedef void *Kuiper_Sparse_valid_smatrix;

typedef struct Kuiper_Sparse_smatrix__uint32_t_s {
    uint32_t nnz1;
    uint32_t *elems1;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_smatrix__uint32_t;

void Kuiper_Sparse_smatrix_id_u32(Kuiper_Sparse_smatrix__uint32_t m);

typedef struct Kuiper_Sparse_sarray__uint32_t_s {
    uint32_t nnz;
    uint32_t len;
    uint32_t *elems;
    uint32_t *pos;
} Kuiper_Sparse_sarray__uint32_t;

void Kuiper_Sparse_sarray_iterator_test_u32(Kuiper_Sparse_sarray__uint32_t a);

#define Kuiper_Sparse_H_DEFINED
#endif                          /* Kuiper_Sparse_H */
