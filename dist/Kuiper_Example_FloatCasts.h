
#ifndef Kuiper_Example_FloatCasts_H
#define Kuiper_Example_FloatCasts_H

#include <kuiper.h>

half Kuiper_Example_FloatCasts_test_cast_f16_f16(half x);

float Kuiper_Example_FloatCasts_test_cast_f16_f32(half x);

double Kuiper_Example_FloatCasts_test_cast_f16_f64(half x);

half Kuiper_Example_FloatCasts_test_cast_f32_f16(float x);

float Kuiper_Example_FloatCasts_test_cast_f32_f32(float x);

double Kuiper_Example_FloatCasts_test_cast_f32_f64(float x);

half Kuiper_Example_FloatCasts_test_cast_f64_f16(double x);

float Kuiper_Example_FloatCasts_test_cast_f64_f32(double x);

double Kuiper_Example_FloatCasts_test_cast_f64_f64(double x);

__nv_bfloat16 Kuiper_Example_FloatCasts_test_cast_bf16_bf16(__nv_bfloat16 x);

float Kuiper_Example_FloatCasts_test_cast_bf16_f32(__nv_bfloat16 x);

__nv_bfloat16 Kuiper_Example_FloatCasts_test_cast_f32_bf16(float x);

__nv_bfloat16 Kuiper_Example_FloatCasts_test_cast_f16_bf16(half x);

half Kuiper_Example_FloatCasts_test_cast_bf16_f16(__nv_bfloat16 x);

double Kuiper_Example_FloatCasts_test_cast_bf16_f64(__nv_bfloat16 x);

__nv_bfloat16 Kuiper_Example_FloatCasts_test_cast_f64_bf16(double x);

#define Kuiper_Example_FloatCasts_H_DEFINED
#endif                          /* Kuiper_Example_FloatCasts_H */
