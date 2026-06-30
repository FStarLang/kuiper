
#ifndef Klas_RowSoftmax_H
#define Klas_RowSoftmax_H

#include <kuiper.h>

void Klas_RowSoftmax_row_softmax_rm_f32(uint32_t m, uint32_t n, uint32_t nth,
                                        float *a);

void Klas_RowSoftmax_row_softmax_rm_f64(uint32_t m, uint32_t n, uint32_t nth,
                                        double *a);

#define Klas_RowSoftmax_H_DEFINED
#endif                          /* Klas_RowSoftmax_H */
