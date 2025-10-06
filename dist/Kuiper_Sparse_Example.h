
#ifndef Kuiper_Sparse_Example_H
#define Kuiper_Sparse_Example_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_sarray__uint32_t_s {
    uint32_t nnz;
    uint32_t len;
    uint32_t *elems;
    uint32_t *pos;
} Kuiper_Sparse_sarray__uint32_t;

void Kuiper_Sparse_Example__id_u32(Kuiper_Sparse_sarray__uint32_t a);

void Kuiper_Sparse_Example__scale_u32(uint32_t k,
                                      Kuiper_Sparse_sarray__uint32_t a);

#define Kuiper_Sparse_Example_H_DEFINED
#endif                          /* Kuiper_Sparse_Example_H */
