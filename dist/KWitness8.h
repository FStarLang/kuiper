
#ifndef KWitness8_H
#define KWitness8_H

#include <kuiper.h>

void
KWitness8_matmul_f32(uint32_t m,
                     uint32_t n,
                     uint32_t k,
                     float alpha, float beta, float *gA, float *gB, float *gC);

#define KWitness8_H_DEFINED
#endif                          /* KWitness8_H */
