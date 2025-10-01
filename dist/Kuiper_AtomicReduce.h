

#ifndef Kuiper_AtomicReduce_H
#define Kuiper_AtomicReduce_H

#include <kuiper.h>

uint32_t Kuiper_AtomicReduce_reduce_u32(uint32_t n, uint32_t *a);

uint64_t Kuiper_AtomicReduce_reduce_u64(uint32_t n, uint64_t *a);

float_t Kuiper_AtomicReduce_reduce_f32(uint32_t n, float_t *a);

double_t Kuiper_AtomicReduce_reduce_f64(uint32_t n, double_t *a);


#define Kuiper_AtomicReduce_H_DEFINED
#endif /* Kuiper_AtomicReduce_H */
