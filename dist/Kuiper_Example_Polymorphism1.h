
#ifndef Kuiper_Example_Polymorphism1_H
#define Kuiper_Example_Polymorphism1_H

#include <kuiper.h>

__device__ void Kuiper_Example_Polymorphism1_kswap__uint64_t(
    uint64_t *r1, uint64_t *r2);

__device__ void Kuiper_Example_Polymorphism1_kswap__float(float *r1, float *r2);

__device__ void Kuiper_Example_Polymorphism1_kswap_U64(
    uint64_t *r1, uint64_t *r2);

__device__ void Kuiper_Example_Polymorphism1_kswap_F32(float *r1, float *r2);

void Kuiper_Example_Polymorphism1_swap_U64(uint64_t *r1, uint64_t *r2);

void Kuiper_Example_Polymorphism1_swap_F32(float *r1, float *r2);

#define Kuiper_Example_Polymorphism1_H_DEFINED
#endif /* Kuiper_Example_Polymorphism1_H */
