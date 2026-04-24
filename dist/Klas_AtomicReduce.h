
#ifndef Klas_AtomicReduce_H
#define Klas_AtomicReduce_H

#include <kuiper.h>

uint32_t Klas_AtomicReduce_reduce_u32(uint32_t n, uint32_t * a);

uint64_t Klas_AtomicReduce_reduce_u64(uint32_t n, uint64_t * a);

float Klas_AtomicReduce_reduce_f32(uint32_t n, float *a);

double Klas_AtomicReduce_reduce_f64(uint32_t n, double *a);

#define Klas_AtomicReduce_H_DEFINED
#endif                          /* Klas_AtomicReduce_H */
