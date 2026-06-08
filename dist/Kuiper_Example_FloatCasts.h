
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

#define Kuiper_Example_FloatCasts_H_DEFINED
#endif                          /* Kuiper_Example_FloatCasts_H */
