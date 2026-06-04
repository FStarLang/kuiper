
#include "Kuiper_Example_FloatCasts.h"

half Kuiper_Example_FloatCasts_test_cast_f16_f16(half x)
{
    return x;
}

float Kuiper_Example_FloatCasts_test_cast_f16_f32(half x)
{
    return __half2float(x);
}

double Kuiper_Example_FloatCasts_test_cast_f16_f64(half x)
{
    return (double)__half2float(x);
}

half Kuiper_Example_FloatCasts_test_cast_f32_f16(float x)
{
    return __float2half_rn(x);
}

float Kuiper_Example_FloatCasts_test_cast_f32_f32(float x)
{
    return x;
}

double Kuiper_Example_FloatCasts_test_cast_f32_f64(float x)
{
    return (double)x;
}

half Kuiper_Example_FloatCasts_test_cast_f64_f16(double x)
{
    return __float2half_rn((float)x);
}

float Kuiper_Example_FloatCasts_test_cast_f64_f32(double x)
{
    return (float)x;
}

double Kuiper_Example_FloatCasts_test_cast_f64_f64(double x)
{
    return x;
}
