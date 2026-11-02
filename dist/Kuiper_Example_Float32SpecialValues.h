
#ifndef Kuiper_Example_Float32SpecialValues_H
#define Kuiper_Example_Float32SpecialValues_H

#include <kuiper.h>

void Kuiper_Example_Float32SpecialValues_run(
    uint32_t n, float *inputs, float *outputs);

void Kuiper_Example_Float32SpecialValues_run_exp_f16(
    uint32_t n, half *inputs, half *outputs);

void Kuiper_Example_Float32SpecialValues_run_exp_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs);

void Kuiper_Example_Float32SpecialValues_run_exp_f64(
    uint32_t n, double *inputs, double *outputs);

void Kuiper_Example_Float32SpecialValues_run_subtraction(
    uint32_t n, float *left, float *right, float *outputs);

#define Kuiper_Example_Float32SpecialValues_H_DEFINED
#endif /* Kuiper_Example_Float32SpecialValues_H */
