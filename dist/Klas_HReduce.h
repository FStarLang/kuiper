
#ifndef Klas_HReduce_H
#define Klas_HReduce_H

#include <kuiper.h>

half Klas_HReduce_reduce_f16_plus(uint32_t nth, uint32_t lena, half * a);

float Klas_HReduce_reduce_f32_plus(uint32_t nth, uint32_t lena, float *a);

double Klas_HReduce_reduce_f64_plus(uint32_t nth, uint32_t lena, double *a);

uint32_t Klas_HReduce_reduce_u32_plus(uint32_t nth, uint32_t lena,
                                      uint32_t * a);

uint64_t Klas_HReduce_reduce_u64_plus(uint32_t nth, uint32_t lena,
                                      uint64_t * a);

#define Klas_HReduce_H_DEFINED
#endif                          /* Klas_HReduce_H */
