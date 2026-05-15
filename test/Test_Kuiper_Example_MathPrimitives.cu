/*
 * Smoke test for math primitives: calls every extracted function
 * and compares the result against the C standard library.
 */

#include "Kuiper_Example_MathPrimitives.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

static int n_tests = 0;
static int n_pass = 0;

static void check_f32(const char *name, float got, float expected, float tol)
{
    n_tests++;
    float err = fabsf(got - expected);
    int ok = (err <= tol) || (isnan(got) && isnan(expected)) || (isinf(got)
                                                                 && isinf(expected)
                                                                 && got == expected);
    if (ok) {
        n_pass++;
    } else {
        printf("FAIL f32 %-12s  got=%e  expected=%e  err=%e\n", name, got, expected, err);
    }
}

static void check_f64(const char *name, double got, double expected, double tol)
{
    n_tests++;
    double err = fabs(got - expected);
    int ok = (err <= tol) || (isnan(got) && isnan(expected)) || (isinf(got)
                                                                 && isinf(expected)
                                                                 && got == expected);
    if (ok) {
        n_pass++;
    } else {
        printf("FAIL f64 %-12s  got=%e  expected=%e  err=%e\n", name, got, expected, err);
    }
}

static void check_f16(const char *name, half got, float expected, float tol)
{
    n_tests++;
    float g = __half2float(got);
    float err = fabsf(g - expected);
    int ok = (err <= tol) || (isnan(g) && isnan(expected)) || (isinf(g)
                                                               && isinf(expected)
                                                               && g == expected);
    if (ok) {
        n_pass++;
    } else {
        printf("FAIL f16 %-12s  got=%e  expected=%e  err=%e\n", name, g, expected, err);
    }
}

#define P "Kuiper_Example_MathPrimitives_"

int main()
{
    /* ---- Float32 ---- */
    float x32 = 0.5f;
    float y32 = 2.0f;
    float z32 = 3.0f;
    float tol32 = 1e-5f;

    /* unary */
    check_f32("sqrt", Kuiper_Example_MathPrimitives_test_sqrt_f32(x32), sqrtf(x32), tol32);
    check_f32("rsqrt", Kuiper_Example_MathPrimitives_test_rsqrt_f32(x32), 1.0f / sqrtf(x32), tol32);
    check_f32("sin", Kuiper_Example_MathPrimitives_test_sin_f32(x32), sinf(x32), tol32);
    check_f32("cos", Kuiper_Example_MathPrimitives_test_cos_f32(x32), cosf(x32), tol32);
    check_f32("tan", Kuiper_Example_MathPrimitives_test_tan_f32(x32), tanf(x32), tol32);
    check_f32("asin", Kuiper_Example_MathPrimitives_test_asin_f32(x32), asinf(x32), tol32);
    check_f32("acos", Kuiper_Example_MathPrimitives_test_acos_f32(x32), acosf(x32), tol32);
    check_f32("atan", Kuiper_Example_MathPrimitives_test_atan_f32(x32), atanf(x32), tol32);
    check_f32("sinh", Kuiper_Example_MathPrimitives_test_sinh_f32(x32), sinhf(x32), tol32);
    check_f32("cosh", Kuiper_Example_MathPrimitives_test_cosh_f32(x32), coshf(x32), tol32);
    check_f32("tanh", Kuiper_Example_MathPrimitives_test_tanh_f32(x32), tanhf(x32), tol32);
    check_f32("ceil", Kuiper_Example_MathPrimitives_test_ceil_f32(x32), ceilf(x32), tol32);
    check_f32("floor", Kuiper_Example_MathPrimitives_test_floor_f32(x32), floorf(x32), tol32);
    check_f32("round", Kuiper_Example_MathPrimitives_test_round_f32(x32), roundf(x32), tol32);
    check_f32("fabs", Kuiper_Example_MathPrimitives_test_fabs_f32(-x32), fabsf(x32), tol32);
    check_f32("erf", Kuiper_Example_MathPrimitives_test_erf_f32(x32), erff(x32), tol32);
    check_f32("log2", Kuiper_Example_MathPrimitives_test_log2_f32(x32), log2f(x32), tol32);
    check_f32("log10", Kuiper_Example_MathPrimitives_test_log10_f32(x32), log10f(x32), tol32);
    check_f32("exp2", Kuiper_Example_MathPrimitives_test_exp2_f32(x32), exp2f(x32), tol32);

    /* binary */
    check_f32("pow", Kuiper_Example_MathPrimitives_test_pow_f32(x32, y32), powf(x32, y32), tol32);
    check_f32("atan2", Kuiper_Example_MathPrimitives_test_atan2_f32(x32, y32),
              atan2f(x32, y32), tol32);
    check_f32("fmin", Kuiper_Example_MathPrimitives_test_fmin_f32(x32, y32),
              fminf(x32, y32), tol32);
    check_f32("fmax", Kuiper_Example_MathPrimitives_test_fmax_f32(x32, y32),
              fmaxf(x32, y32), tol32);
    check_f32("fmod", Kuiper_Example_MathPrimitives_test_fmod_f32(y32, x32),
              fmodf(y32, x32), tol32);
    check_f32("copysign",
              Kuiper_Example_MathPrimitives_test_copysign_f32(x32, -y32),
              copysignf(x32, -y32), tol32);

    /* ternary */
    check_f32("fma", Kuiper_Example_MathPrimitives_test_fma_f32(x32, y32, z32),
              fmaf(x32, y32, z32), tol32);

    /* ---- Float64 ---- */
    double x64 = 0.5;
    double y64 = 2.0;
    double z64 = 3.0;
    double tol64 = 1e-12;

    check_f64("sqrt", Kuiper_Example_MathPrimitives_test_sqrt_f64(x64), sqrt(x64), tol64);
    check_f64("rsqrt", Kuiper_Example_MathPrimitives_test_rsqrt_f64(x64), 1.0 / sqrt(x64), tol64);
    check_f64("sin", Kuiper_Example_MathPrimitives_test_sin_f64(x64), sin(x64), tol64);
    check_f64("cos", Kuiper_Example_MathPrimitives_test_cos_f64(x64), cos(x64), tol64);
    check_f64("tan", Kuiper_Example_MathPrimitives_test_tan_f64(x64), tan(x64), tol64);
    check_f64("asin", Kuiper_Example_MathPrimitives_test_asin_f64(x64), asin(x64), tol64);
    check_f64("acos", Kuiper_Example_MathPrimitives_test_acos_f64(x64), acos(x64), tol64);
    check_f64("atan", Kuiper_Example_MathPrimitives_test_atan_f64(x64), atan(x64), tol64);
    check_f64("sinh", Kuiper_Example_MathPrimitives_test_sinh_f64(x64), sinh(x64), tol64);
    check_f64("cosh", Kuiper_Example_MathPrimitives_test_cosh_f64(x64), cosh(x64), tol64);
    check_f64("tanh", Kuiper_Example_MathPrimitives_test_tanh_f64(x64), tanh(x64), tol64);
    check_f64("ceil", Kuiper_Example_MathPrimitives_test_ceil_f64(x64), ceil(x64), tol64);
    check_f64("floor", Kuiper_Example_MathPrimitives_test_floor_f64(x64), floor(x64), tol64);
    check_f64("round", Kuiper_Example_MathPrimitives_test_round_f64(x64), round(x64), tol64);
    check_f64("fabs", Kuiper_Example_MathPrimitives_test_fabs_f64(-x64), fabs(x64), tol64);
    check_f64("erf", Kuiper_Example_MathPrimitives_test_erf_f64(x64), erf(x64), tol64);
    check_f64("log2", Kuiper_Example_MathPrimitives_test_log2_f64(x64), log2(x64), tol64);
    check_f64("log10", Kuiper_Example_MathPrimitives_test_log10_f64(x64), log10(x64), tol64);
    check_f64("exp2", Kuiper_Example_MathPrimitives_test_exp2_f64(x64), exp2(x64), tol64);
    check_f64("pow", Kuiper_Example_MathPrimitives_test_pow_f64(x64, y64), pow(x64, y64), tol64);
    check_f64("atan2", Kuiper_Example_MathPrimitives_test_atan2_f64(x64, y64),
              atan2(x64, y64), tol64);
    check_f64("fmin", Kuiper_Example_MathPrimitives_test_fmin_f64(x64, y64), fmin(x64, y64), tol64);
    check_f64("fmax", Kuiper_Example_MathPrimitives_test_fmax_f64(x64, y64), fmax(x64, y64), tol64);
    check_f64("fmod", Kuiper_Example_MathPrimitives_test_fmod_f64(y64, x64), fmod(y64, x64), tol64);
    check_f64("copysign",
              Kuiper_Example_MathPrimitives_test_copysign_f64(x64, -y64),
              copysign(x64, -y64), tol64);
    check_f64("fma", Kuiper_Example_MathPrimitives_test_fma_f64(x64, y64, z64),
              fma(x64, y64, z64), tol64);

    /* ---- Float16 ---- */
    half x16 = __float2half(0.5f);
    half y16 = __float2half(2.0f);
    half z16 = __float2half(3.0f);
    float tol16 = 2e-2f;        /* f16 has ~3 decimal digits of precision */

    check_f16("sqrt", Kuiper_Example_MathPrimitives_test_sqrt_f16(x16), sqrtf(0.5f), tol16);
    check_f16("rsqrt", Kuiper_Example_MathPrimitives_test_rsqrt_f16(x16),
              1.0f / sqrtf(0.5f), tol16);
    check_f16("sin", Kuiper_Example_MathPrimitives_test_sin_f16(x16), sinf(0.5f), tol16);
    check_f16("cos", Kuiper_Example_MathPrimitives_test_cos_f16(x16), cosf(0.5f), tol16);
    check_f16("tan", Kuiper_Example_MathPrimitives_test_tan_f16(x16), tanf(0.5f), tol16);
    check_f16("asin", Kuiper_Example_MathPrimitives_test_asin_f16(x16), asinf(0.5f), tol16);
    check_f16("acos", Kuiper_Example_MathPrimitives_test_acos_f16(x16), acosf(0.5f), tol16);
    check_f16("atan", Kuiper_Example_MathPrimitives_test_atan_f16(x16), atanf(0.5f), tol16);
    check_f16("sinh", Kuiper_Example_MathPrimitives_test_sinh_f16(x16), sinhf(0.5f), tol16);
    check_f16("cosh", Kuiper_Example_MathPrimitives_test_cosh_f16(x16), coshf(0.5f), tol16);
    check_f16("tanh", Kuiper_Example_MathPrimitives_test_tanh_f16(x16), tanhf(0.5f), tol16);
    check_f16("ceil", Kuiper_Example_MathPrimitives_test_ceil_f16(x16), ceilf(0.5f), tol16);
    check_f16("floor", Kuiper_Example_MathPrimitives_test_floor_f16(x16), floorf(0.5f), tol16);
    check_f16("round", Kuiper_Example_MathPrimitives_test_round_f16(x16), roundf(0.5f), tol16);
    check_f16("fabs", Kuiper_Example_MathPrimitives_test_fabs_f16(__float2half(-0.5f)), fabsf(0.5f),
              tol16);
    check_f16("erf", Kuiper_Example_MathPrimitives_test_erf_f16(x16), erff(0.5f), tol16);
    check_f16("log2", Kuiper_Example_MathPrimitives_test_log2_f16(x16), log2f(0.5f), tol16);
    check_f16("log10", Kuiper_Example_MathPrimitives_test_log10_f16(x16), log10f(0.5f), tol16);
    check_f16("exp2", Kuiper_Example_MathPrimitives_test_exp2_f16(x16), exp2f(0.5f), tol16);
    check_f16("pow", Kuiper_Example_MathPrimitives_test_pow_f16(x16, y16), powf(0.5f, 2.0f), tol16);
    check_f16("atan2", Kuiper_Example_MathPrimitives_test_atan2_f16(x16, y16),
              atan2f(0.5f, 2.0f), tol16);
    check_f16("fmin", Kuiper_Example_MathPrimitives_test_fmin_f16(x16, y16),
              fminf(0.5f, 2.0f), tol16);
    check_f16("fmax", Kuiper_Example_MathPrimitives_test_fmax_f16(x16, y16),
              fmaxf(0.5f, 2.0f), tol16);
    check_f16("fmod", Kuiper_Example_MathPrimitives_test_fmod_f16(y16, x16),
              fmodf(2.0f, 0.5f), tol16);
    check_f16("copysign",
              Kuiper_Example_MathPrimitives_test_copysign_f16(x16, __float2half(-2.0f)),
              copysignf(0.5f, -2.0f), tol16);
    check_f16("fma", Kuiper_Example_MathPrimitives_test_fma_f16(x16, y16, z16),
              fmaf(0.5f, 2.0f, 3.0f), tol16);

    /* ---- valid / min_val / max_val ---- */

    /* Float32 */
    n_tests++;
    if (Kuiper_Example_MathPrimitives_test_valid_f32(1.0f))
        n_pass++;
    else
        printf("FAIL f32 valid(1.0f) returned false\n");

    n_tests++;
    if (!Kuiper_Example_MathPrimitives_test_valid_f32(NAN))
        n_pass++;
    else
        printf("FAIL f32 valid(NAN) returned true\n");

    n_tests++;
    if (!Kuiper_Example_MathPrimitives_test_valid_f32(INFINITY))
        n_pass++;
    else
        printf("FAIL f32 valid(INFINITY) returned true\n");

    check_f32("min_val", Kuiper_Example_MathPrimitives_test_min_val_f32(), -FLT_MAX, 0.0f);
    check_f32("max_val", Kuiper_Example_MathPrimitives_test_max_val_f32(), FLT_MAX, 0.0f);

    /* Float64 */
    n_tests++;
    if (Kuiper_Example_MathPrimitives_test_valid_f64(1.0))
        n_pass++;
    else
        printf("FAIL f64 valid(1.0) returned false\n");

    n_tests++;
    if (!Kuiper_Example_MathPrimitives_test_valid_f64((double)NAN))
        n_pass++;
    else
        printf("FAIL f64 valid(NAN) returned true\n");

    n_tests++;
    if (!Kuiper_Example_MathPrimitives_test_valid_f64((double)INFINITY))
        n_pass++;
    else
        printf("FAIL f64 valid(INFINITY) returned true\n");

    check_f64("min_val", Kuiper_Example_MathPrimitives_test_min_val_f64(), -DBL_MAX, 0.0);
    check_f64("max_val", Kuiper_Example_MathPrimitives_test_max_val_f64(), DBL_MAX, 0.0);

    /* Float16 */
    n_tests++;
    if (Kuiper_Example_MathPrimitives_test_valid_f16(__float2half(1.0f)))
        n_pass++;
    else
        printf("FAIL f16 valid(1.0) returned false\n");

    n_tests++;
    if (!Kuiper_Example_MathPrimitives_test_valid_f16(__float2half(NAN)))
        n_pass++;
    else
        printf("FAIL f16 valid(NAN) returned true\n");

    n_tests++;
    if (!Kuiper_Example_MathPrimitives_test_valid_f16(__float2half(INFINITY)))
        n_pass++;
    else
        printf("FAIL f16 valid(INFINITY) returned true\n");

    check_f16("min_val", Kuiper_Example_MathPrimitives_test_min_val_f16(), -65504.0f, 0.0f);
    check_f16("max_val", Kuiper_Example_MathPrimitives_test_max_val_f16(), 65504.0f, 0.0f);

    if (n_pass == n_tests) {
        printf("%d tests, OK\n", n_tests);
    } else {
        printf("%d/%d tests FAILED\n", n_tests - n_pass, n_tests);
    }

    return n_pass != n_tests;
}
