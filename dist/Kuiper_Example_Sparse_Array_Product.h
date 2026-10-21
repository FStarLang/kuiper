
#ifndef Kuiper_Example_Sparse_Array_Product_H
#define Kuiper_Example_Sparse_Array_Product_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_Array_sarray__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *pos;
} Kuiper_Sparse_Array_sarray__uint32_t;

uint32_t Kuiper_Example_Sparse_Array_Product_sarray_product_dense_u32(
    Kuiper_Sparse_Array_sarray__uint32_t a, uint32_t *v);

#define Kuiper_Example_Sparse_Array_Product_H_DEFINED
#endif /* Kuiper_Example_Sparse_Array_Product_H */
