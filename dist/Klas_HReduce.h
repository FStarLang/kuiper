
#ifndef Klas_HReduce_H
#define Klas_HReduce_H

#include <kuiper.h>

void Klas_HReduce_reduce_f16_plus(uint32_t lena, half * a);

void Klas_HReduce_reduce_f32_plus(uint32_t lena, float *a);

void Klas_HReduce_reduce_f64_plus(uint32_t lena, double *a);

void Klas_HReduce_reduce_u32_plus(uint32_t lena, uint32_t * a);

void Klas_HReduce_reduce_u64_plus(uint32_t lena, uint64_t * a);

#define Klas_HReduce_H_DEFINED
#endif                          /* Klas_HReduce_H */
