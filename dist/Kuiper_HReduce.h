
#ifndef Kuiper_HReduce_H
#define Kuiper_HReduce_H

#include <kuiper.h>

void Kuiper_HReduce_reduce_f16_plus(uint32_t lena, half * a);

void Kuiper_HReduce_reduce_f32_plus(uint32_t lena, float *a);

void Kuiper_HReduce_reduce_f64_plus(uint32_t lena, double *a);

void Kuiper_HReduce_reduce_u32_plus(uint32_t lena, uint32_t * a);

void Kuiper_HReduce_reduce_u64_plus(uint32_t lena, uint64_t * a);

#define Kuiper_HReduce_H_DEFINED
#endif                          /* Kuiper_HReduce_H */
