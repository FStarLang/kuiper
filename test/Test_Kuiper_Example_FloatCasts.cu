/*
 * Smoke test for float casts: calls every extracted cast variant and
 * compares the result against the host conversion.
 */

#include "Kuiper_Example_FloatCasts.h"
#include <math.h>
#include <stdio.h>

static int n_tests = 0;
static int n_pass = 0;

static void check(const char *name, double got, double expected, double tol)
{
    n_tests++;
    double err = fabs(got - expected);
    int ok = (err <= tol) || (isnan(got) && isnan(expected))
        || (isinf(got) && isinf(expected) && got == expected);
    if (ok) {
        n_pass++;
    } else {
        printf("FAIL %-12s  got=%e  expected=%e  err=%e\n", name, got, expected, err);
    }
}

int main()
{
    float xf = 1.5f;
    double xd = 1.5;
    half xh = __float2half(1.5f);

    /* from f16 */
    check("f16->f16", __half2float(Kuiper_Example_FloatCasts_test_cast_f16_f16(xh)),
          __half2float(xh), 0.0);
    check("f16->f32", Kuiper_Example_FloatCasts_test_cast_f16_f32(xh), __half2float(xh), 0.0);
    check("f16->f64", Kuiper_Example_FloatCasts_test_cast_f16_f64(xh),
          (double)__half2float(xh), 0.0);

    /* from f32 */
    check("f32->f16", __half2float(Kuiper_Example_FloatCasts_test_cast_f32_f16(xf)),
          __half2float(__float2half_rn(xf)), 0.0);
    check("f32->f32", Kuiper_Example_FloatCasts_test_cast_f32_f32(xf), (double)xf, 0.0);
    check("f32->f64", Kuiper_Example_FloatCasts_test_cast_f32_f64(xf), (double)xf, 0.0);

    /* from f64 */
    check("f64->f16", __half2float(Kuiper_Example_FloatCasts_test_cast_f64_f16(xd)),
          __half2float(__float2half_rn((float)xd)), 0.0);
    check("f64->f32", Kuiper_Example_FloatCasts_test_cast_f64_f32(xd), (double)(float)xd, 0.0);
    check("f64->f64", Kuiper_Example_FloatCasts_test_cast_f64_f64(xd), xd, 0.0);

    if (n_pass == n_tests) {
        printf("%d tests, OK\n", n_tests);
    } else {
        printf("%d/%d tests FAILED\n", n_tests - n_pass, n_tests);
    }

    return n_pass != n_tests;
}
