
#ifndef Kuiper_Example_Float32GPU_H
#define Kuiper_Example_Float32GPU_H

#include <kuiper.h>

float Kuiper_Example_Float32GPU_multiply(float x, float y);

float Kuiper_Example_Float32GPU_exponentiate(float x);

float Kuiper_Example_Float32GPU_reciprocal(float x);

float Kuiper_Example_Float32GPU_inverse_root(float x);

void Kuiper_Example_Float32GPU_arithmetic(
    uint32_t n, float *inputs, float *outputs);

#define Kuiper_Example_Float32GPU_H_DEFINED
#endif /* Kuiper_Example_Float32GPU_H */
