

#ifndef Kuiper_Sparse_H
#define Kuiper_Sparse_H

#include <kuiper.h>

extern uint32_t Kuiper_Sparse_x;

typedef struct Kuiper_Sparse_sarray_iterator_s
{
  size_t i;
  size_t pos1;
}
Kuiper_Sparse_sarray_iterator;

typedef void *Kuiper_Sparse_valid_smatrix;


#define Kuiper_Sparse_H_DEFINED
#endif /* Kuiper_Sparse_H */
