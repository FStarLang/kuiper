

#ifndef __Kuiper_AtomicReduce_H
#define __Kuiper_AtomicReduce_H

#include <kuiper.h>

uint32_t Kuiper_AtomicReduce_reduce_u32(size_t n, uint32_t *a);

uint64_t Kuiper_AtomicReduce_reduce_u64(size_t n, uint64_t *a);

float_t Kuiper_AtomicReduce_reduce_f32(size_t n, float_t *a);

double_t Kuiper_AtomicReduce_reduce_f64(size_t n, double_t *a);


#define __Kuiper_AtomicReduce_H_DEFINED
#endif
