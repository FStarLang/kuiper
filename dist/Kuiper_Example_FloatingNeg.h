
#ifndef Kuiper_Example_FloatingNeg_H
#define Kuiper_Example_FloatingNeg_H

#include <kuiper.h>

half Kuiper_Example_FloatingNeg_host_f16(half x);

__nv_bfloat16 Kuiper_Example_FloatingNeg_host_bf16(__nv_bfloat16 x);

float Kuiper_Example_FloatingNeg_host_f32(float x);

double Kuiper_Example_FloatingNeg_host_f64(double x);

half Kuiper_Example_FloatingNeg_host_double_neg_f16(half x);

__nv_bfloat16 Kuiper_Example_FloatingNeg_host_double_neg_bf16(__nv_bfloat16 x);

float Kuiper_Example_FloatingNeg_host_double_neg_f32(float x);

double Kuiper_Example_FloatingNeg_host_double_neg_f64(double x);

half Kuiper_Example_FloatingNeg_host_zero_sub_f16(half x);

__nv_bfloat16 Kuiper_Example_FloatingNeg_host_zero_sub_bf16(__nv_bfloat16 x);

float Kuiper_Example_FloatingNeg_host_zero_sub_f32(float x);

double Kuiper_Example_FloatingNeg_host_zero_sub_f64(double x);

void Kuiper_Example_FloatingNeg_run_f16(
    uint32_t n, half *inputs, half *outputs);

void Kuiper_Example_FloatingNeg_run_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs);

void Kuiper_Example_FloatingNeg_run_f32(
    uint32_t n, float *inputs, float *outputs);

void Kuiper_Example_FloatingNeg_run_f64(
    uint32_t n, double *inputs, double *outputs);

void Kuiper_Example_FloatingNeg_run_double_neg_f16(
    uint32_t n, half *inputs, half *outputs);

void Kuiper_Example_FloatingNeg_run_double_neg_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs);

void Kuiper_Example_FloatingNeg_run_double_neg_f32(
    uint32_t n, float *inputs, float *outputs);

void Kuiper_Example_FloatingNeg_run_double_neg_f64(
    uint32_t n, double *inputs, double *outputs);

void Kuiper_Example_FloatingNeg_run_zero_sub_f16(
    uint32_t n, half *inputs, half *outputs);

void Kuiper_Example_FloatingNeg_run_zero_sub_bf16(
    uint32_t n, __nv_bfloat16 *inputs, __nv_bfloat16 *outputs);

void Kuiper_Example_FloatingNeg_run_zero_sub_f32(
    uint32_t n, float *inputs, float *outputs);

void Kuiper_Example_FloatingNeg_run_zero_sub_f64(
    uint32_t n, double *inputs, double *outputs);

#define Kuiper_Example_FloatingNeg_H_DEFINED
#endif /* Kuiper_Example_FloatingNeg_H */
