
#ifndef Kuiper_GraphDist_H
#define Kuiper_GraphDist_H

#include <kuiper.h>

__device__ uint16_t Kuiper_GraphDist_add_(uint16_t x, uint16_t y);

__device__ uint16_t Kuiper_GraphDist_mult(uint16_t x, uint16_t y);

typedef uint16_t Kuiper_GraphDist_dist;

void Kuiper_GraphDist_matmul_dist_gpu(uint32_t size, uint16_t *a, uint16_t *b);

#define Kuiper_GraphDist_H_DEFINED
#endif /* Kuiper_GraphDist_H */
