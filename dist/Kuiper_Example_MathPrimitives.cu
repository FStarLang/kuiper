
#include "Kuiper_Example_MathPrimitives.h"

half Kuiper_Example_MathPrimitives_test_sqrt_f16(half x)
{
    return kpr_hsqrt(x);
}

half Kuiper_Example_MathPrimitives_test_rsqrt_f16(half x)
{
    return kpr_hrsqrt(x);
}

half Kuiper_Example_MathPrimitives_test_sin_f16(half x)
{
    return kpr_hsin(x);
}

half Kuiper_Example_MathPrimitives_test_cos_f16(half x)
{
    return kpr_hcos(x);
}

half Kuiper_Example_MathPrimitives_test_tan_f16(half x)
{
    return kpr_htan(x);
}

half Kuiper_Example_MathPrimitives_test_asin_f16(half x)
{
    return kpr_hasin(x);
}

half Kuiper_Example_MathPrimitives_test_acos_f16(half x)
{
    return kpr_hacos(x);
}

half Kuiper_Example_MathPrimitives_test_atan_f16(half x)
{
    return kpr_hatan(x);
}

half Kuiper_Example_MathPrimitives_test_sinh_f16(half x)
{
    return kpr_hsinh(x);
}

half Kuiper_Example_MathPrimitives_test_cosh_f16(half x)
{
    return kpr_hcosh(x);
}

half Kuiper_Example_MathPrimitives_test_tanh_f16(half x)
{
    return kpr_htanh(x);
}

half Kuiper_Example_MathPrimitives_test_ceil_f16(half x)
{
    return kpr_hceil(x);
}

half Kuiper_Example_MathPrimitives_test_floor_f16(half x)
{
    return kpr_hfloor(x);
}

half Kuiper_Example_MathPrimitives_test_round_f16(half x)
{
    return kpr_hround(x);
}

half Kuiper_Example_MathPrimitives_test_fabs_f16(half x)
{
    return kpr_hfabs(x);
}

half Kuiper_Example_MathPrimitives_test_erf_f16(half x)
{
    return kpr_herf(x);
}

half Kuiper_Example_MathPrimitives_test_log2_f16(half x)
{
    return kpr_hlog2(x);
}

half Kuiper_Example_MathPrimitives_test_log10_f16(half x)
{
    return kpr_hlog10(x);
}

half Kuiper_Example_MathPrimitives_test_exp2_f16(half x)
{
    return kpr_hexp2(x);
}

half Kuiper_Example_MathPrimitives_test_pow_f16(half x, half y)
{
    return kpr_hpow(x, y);
}

half Kuiper_Example_MathPrimitives_test_atan2_f16(half x, half y)
{
    return kpr_hatan2(x, y);
}

half Kuiper_Example_MathPrimitives_test_fmin_f16(half x, half y)
{
    return kpr_hfmin(x, y);
}

half Kuiper_Example_MathPrimitives_test_fmax_f16(half x, half y)
{
    return kpr_hfmax(x, y);
}

half Kuiper_Example_MathPrimitives_test_fmod_f16(half x, half y)
{
    return kpr_hfmod(x, y);
}

half Kuiper_Example_MathPrimitives_test_copysign_f16(half x, half y)
{
    return kpr_hcopysign(x, y);
}

half Kuiper_Example_MathPrimitives_test_fma_f16(half x, half y, half z)
{
    return kpr_hfma(x, y, z);
}

half Kuiper_Example_MathPrimitives_test_largest_f16(void)
{
    return HLF_MAX;
}

half Kuiper_Example_MathPrimitives_test_infinity_f16(void)
{
    return HLF_INFINITY;
}

float Kuiper_Example_MathPrimitives_test_sqrt_f32(float x)
{
    return sqrtf(x);
}

float Kuiper_Example_MathPrimitives_test_rsqrt_f32(float x)
{
    return rsqrtf(x);
}

float Kuiper_Example_MathPrimitives_test_sin_f32(float x)
{
    return sinf(x);
}

float Kuiper_Example_MathPrimitives_test_cos_f32(float x)
{
    return cosf(x);
}

float Kuiper_Example_MathPrimitives_test_tan_f32(float x)
{
    return tanf(x);
}

float Kuiper_Example_MathPrimitives_test_asin_f32(float x)
{
    return asinf(x);
}

float Kuiper_Example_MathPrimitives_test_acos_f32(float x)
{
    return acosf(x);
}

float Kuiper_Example_MathPrimitives_test_atan_f32(float x)
{
    return atanf(x);
}

float Kuiper_Example_MathPrimitives_test_sinh_f32(float x)
{
    return sinhf(x);
}

float Kuiper_Example_MathPrimitives_test_cosh_f32(float x)
{
    return coshf(x);
}

float Kuiper_Example_MathPrimitives_test_tanh_f32(float x)
{
    return tanhf(x);
}

float Kuiper_Example_MathPrimitives_test_ceil_f32(float x)
{
    return ceilf(x);
}

float Kuiper_Example_MathPrimitives_test_floor_f32(float x)
{
    return floorf(x);
}

float Kuiper_Example_MathPrimitives_test_round_f32(float x)
{
    return roundf(x);
}

float Kuiper_Example_MathPrimitives_test_fabs_f32(float x)
{
    return fabsf(x);
}

float Kuiper_Example_MathPrimitives_test_erf_f32(float x)
{
    return erff(x);
}

float Kuiper_Example_MathPrimitives_test_log2_f32(float x)
{
    return log2f(x);
}

float Kuiper_Example_MathPrimitives_test_log10_f32(float x)
{
    return log10f(x);
}

float Kuiper_Example_MathPrimitives_test_exp2_f32(float x)
{
    return exp2f(x);
}

float Kuiper_Example_MathPrimitives_test_pow_f32(float x, float y)
{
    return powf(x, y);
}

float Kuiper_Example_MathPrimitives_test_atan2_f32(float x, float y)
{
    return atan2f(x, y);
}

float Kuiper_Example_MathPrimitives_test_fmin_f32(float x, float y)
{
    return fminf(x, y);
}

float Kuiper_Example_MathPrimitives_test_fmax_f32(float x, float y)
{
    return fmaxf(x, y);
}

float Kuiper_Example_MathPrimitives_test_fmod_f32(float x, float y)
{
    return fmodf(x, y);
}

float Kuiper_Example_MathPrimitives_test_copysign_f32(float x, float y)
{
    return copysignf(x, y);
}

float Kuiper_Example_MathPrimitives_test_fma_f32(float x, float y, float z)
{
    return fmaf(x, y, z);
}

float Kuiper_Example_MathPrimitives_test_largest_f32(void)
{
    return FLT_MAX;
}

float Kuiper_Example_MathPrimitives_test_infinity_f32(void)
{
    return INFINITY;
}

double Kuiper_Example_MathPrimitives_test_sqrt_f64(double x)
{
    return sqrt(x);
}

double Kuiper_Example_MathPrimitives_test_rsqrt_f64(double x)
{
    return rsqrt(x);
}

double Kuiper_Example_MathPrimitives_test_sin_f64(double x)
{
    return sin(x);
}

double Kuiper_Example_MathPrimitives_test_cos_f64(double x)
{
    return cos(x);
}

double Kuiper_Example_MathPrimitives_test_tan_f64(double x)
{
    return tan(x);
}

double Kuiper_Example_MathPrimitives_test_asin_f64(double x)
{
    return asin(x);
}

double Kuiper_Example_MathPrimitives_test_acos_f64(double x)
{
    return acos(x);
}

double Kuiper_Example_MathPrimitives_test_atan_f64(double x)
{
    return atan(x);
}

double Kuiper_Example_MathPrimitives_test_sinh_f64(double x)
{
    return sinh(x);
}

double Kuiper_Example_MathPrimitives_test_cosh_f64(double x)
{
    return cosh(x);
}

double Kuiper_Example_MathPrimitives_test_tanh_f64(double x)
{
    return tanh(x);
}

double Kuiper_Example_MathPrimitives_test_ceil_f64(double x)
{
    return ceil(x);
}

double Kuiper_Example_MathPrimitives_test_floor_f64(double x)
{
    return floor(x);
}

double Kuiper_Example_MathPrimitives_test_round_f64(double x)
{
    return round(x);
}

double Kuiper_Example_MathPrimitives_test_fabs_f64(double x)
{
    return fabs(x);
}

double Kuiper_Example_MathPrimitives_test_erf_f64(double x)
{
    return erf(x);
}

double Kuiper_Example_MathPrimitives_test_log2_f64(double x)
{
    return log2(x);
}

double Kuiper_Example_MathPrimitives_test_log10_f64(double x)
{
    return log10(x);
}

double Kuiper_Example_MathPrimitives_test_exp2_f64(double x)
{
    return exp2(x);
}

double Kuiper_Example_MathPrimitives_test_pow_f64(double x, double y)
{
    return pow(x, y);
}

double Kuiper_Example_MathPrimitives_test_atan2_f64(double x, double y)
{
    return atan2(x, y);
}

double Kuiper_Example_MathPrimitives_test_fmin_f64(double x, double y)
{
    return fmin(x, y);
}

double Kuiper_Example_MathPrimitives_test_fmax_f64(double x, double y)
{
    return fmax(x, y);
}

double Kuiper_Example_MathPrimitives_test_fmod_f64(double x, double y)
{
    return fmod(x, y);
}

double Kuiper_Example_MathPrimitives_test_copysign_f64(double x, double y)
{
    return copysign(x, y);
}

double Kuiper_Example_MathPrimitives_test_fma_f64(double x, double y, double z)
{
    return fma(x, y, z);
}

double Kuiper_Example_MathPrimitives_test_largest_f64(void)
{
    return DBL_MAX;
}

double Kuiper_Example_MathPrimitives_test_infinity_f64(void)
{
    return INFINITY;
}
