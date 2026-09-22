
#include "Kuiper_Example_FloatEquality.h"

bool Kuiper_Example_FloatEquality_ieee_f16(half x, half y) { return x == y; }

bool Kuiper_Example_FloatEquality_bits_f16(half x, half y)
{
    return kpr_f16_bit_eq(x, y);
}

bool Kuiper_Example_FloatEquality_eq_f16(half x, half y) { return x == y; }

bool Kuiper_Example_FloatEquality_ieee_bf16(__nv_bfloat16 x, __nv_bfloat16 y)
{
    return x == y;
}

bool Kuiper_Example_FloatEquality_bits_bf16(__nv_bfloat16 x, __nv_bfloat16 y)
{
    return kpr_bf16_bit_eq(x, y);
}

bool Kuiper_Example_FloatEquality_eq_bf16(__nv_bfloat16 x, __nv_bfloat16 y)
{
    return x == y;
}

bool Kuiper_Example_FloatEquality_ieee_f32(float x, float y) { return x == y; }

bool Kuiper_Example_FloatEquality_bits_f32(float x, float y)
{
    return kpr_f32_bit_eq(x, y);
}

bool Kuiper_Example_FloatEquality_eq_f32(float x, float y) { return x == y; }

float Kuiper_Example_FloatEquality_mul_zero_f32(float x)
{
    return x * (float) 0LL;
}

float Kuiper_Example_FloatEquality_add_zero_f32(float x)
{
    return x + (float) 0LL;
}

float Kuiper_Example_FloatEquality_reciprocal_f32(float x)
{
    return (float) 1LL / x;
}

bool Kuiper_Example_FloatEquality_ieee_f64(double x, double y)
{
    return x == y;
}

bool Kuiper_Example_FloatEquality_bits_f64(double x, double y)
{
    return kpr_f64_bit_eq(x, y);
}

bool Kuiper_Example_FloatEquality_eq_f64(double x, double y) { return x == y; }

double Kuiper_Example_FloatEquality_mul_zero_f64(double x)
{
    return x * (double) 0LL;
}

double Kuiper_Example_FloatEquality_add_zero_f64(double x)
{
    return x + (double) 0LL;
}

double Kuiper_Example_FloatEquality_reciprocal_f64(double x)
{
    return (double) 1LL / x;
}
