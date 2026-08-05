
#ifndef KWitness2_H
#define KWitness2_H

#include <kuiper.h>

void KWitness2_matmul_f32(uint32_t m, uint32_t n, uint32_t k, float *gA,
                          float *gB, float *gC);

#define KWitness2_H_DEFINED
#endif                          /* KWitness2_H */
