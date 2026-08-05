
#ifndef KWitness4_H
#define KWitness4_H

#include <kuiper.h>

void KWitness4_matmul_f32(uint32_t m, uint32_t n, uint32_t k, float *gA,
                          float *gB, float *gC);

#define KWitness4_H_DEFINED
#endif                          /* KWitness4_H */
