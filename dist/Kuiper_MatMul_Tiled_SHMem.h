

#ifndef __Kuiper_MatMul_Tiled_SHMem_H
#define __Kuiper_MatMul_Tiled_SHMem_H


#include <kuiper.h>

void Kuiper_MatMul_Tiled_SHMem_inst_gpu(uint64_t *gA, uint64_t *gB, uint64_t *gC);

uint64_t *Kuiper_MatMul_Tiled_SHMem_matmul(uint64_t *a, uint64_t *b);


#define __Kuiper_MatMul_Tiled_SHMem_H_DEFINED
#endif
