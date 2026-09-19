
#ifndef Kuiper_Example_FloatEquality_H
#define Kuiper_Example_FloatEquality_H

#include <kuiper.h>

bool Kuiper_Example_FloatEquality_ieee_f16(half x, half y);

bool Kuiper_Example_FloatEquality_bits_f16(half x, half y);

bool Kuiper_Example_FloatEquality_eq_f16(half x, half y);

bool Kuiper_Example_FloatEquality_ieee_bf16(__nv_bfloat16 x, __nv_bfloat16 y);

bool Kuiper_Example_FloatEquality_bits_bf16(__nv_bfloat16 x, __nv_bfloat16 y);

bool Kuiper_Example_FloatEquality_eq_bf16(__nv_bfloat16 x, __nv_bfloat16 y);

bool Kuiper_Example_FloatEquality_ieee_f32(float x, float y);

bool Kuiper_Example_FloatEquality_bits_f32(float x, float y);

bool Kuiper_Example_FloatEquality_eq_f32(float x, float y);

float Kuiper_Example_FloatEquality_mul_zero_f32(float x);

float Kuiper_Example_FloatEquality_add_zero_f32(float x);

float Kuiper_Example_FloatEquality_reciprocal_f32(float x);

bool Kuiper_Example_FloatEquality_ieee_f64(double x, double y);

bool Kuiper_Example_FloatEquality_bits_f64(double x, double y);

bool Kuiper_Example_FloatEquality_eq_f64(double x, double y);

double Kuiper_Example_FloatEquality_mul_zero_f64(double x);

double Kuiper_Example_FloatEquality_add_zero_f64(double x);

double Kuiper_Example_FloatEquality_reciprocal_f64(double x);

#define Kuiper_Example_FloatEquality_H_DEFINED
#endif /* Kuiper_Example_FloatEquality_H */
