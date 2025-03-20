

#ifndef __Kuiper_GraphDist_H
#define __Kuiper_GraphDist_H

#include <kuiper.h>

typedef uint16_t Kuiper_GraphDist_dist;

bool Kuiper_GraphDist_uu___is_D(uint16_t projectee);

__device__

uint16_t Kuiper_GraphDist_add(uint16_t x, uint16_t y);

__device__

uint16_t Kuiper_GraphDist_add_(uint16_t x, uint16_t y);

__device__

uint16_t Kuiper_GraphDist_mult(uint16_t x, uint16_t y);

void Kuiper_GraphDist_matmul_dist_gpu(size_t size, uint16_t *a, uint16_t *b);


#define __Kuiper_GraphDist_H_DEFINED
#endif
