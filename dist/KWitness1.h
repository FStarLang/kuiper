
#ifndef KWitness1_H
#define KWitness1_H

#include <kuiper.h>

void KWitness1_matmul_f32(uint32_t m, uint32_t n, uint32_t k, float *gA,
                          float *gB, float *gC);

#define KWitness1_H_DEFINED
#endif                          /* KWitness1_H */
