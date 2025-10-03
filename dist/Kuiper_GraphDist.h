
#ifndef Kuiper_GraphDist_H
#define Kuiper_GraphDist_H

#include <kuiper.h>

typedef uint16_t Kuiper_GraphDist_dist;

bool Kuiper_GraphDist_uu___is_D(uint16_t projectee);

void Kuiper_GraphDist_matmul_dist_gpu(uint32_t size, uint16_t * a,
                                      uint16_t * b);

#define Kuiper_GraphDist_H_DEFINED
#endif                          /* Kuiper_GraphDist_H */
