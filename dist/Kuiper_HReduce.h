

#ifndef Kuiper_HReduce_H
#define Kuiper_HReduce_H

#include <kuiper.h>

void Kuiper_HReduce_reduce_f16_plus(size_t lena, half_t *a);

void Kuiper_HReduce_reduce_f32_plus(size_t lena, float_t *a);

void Kuiper_HReduce_reduce_f64_plus(size_t lena, double_t *a);

void Kuiper_HReduce_reduce_u32_plus(size_t lena, uint32_t *a);

void Kuiper_HReduce_reduce_u64_plus(size_t lena, uint64_t *a);


#define Kuiper_HReduce_H_DEFINED
#endif /* Kuiper_HReduce_H */
