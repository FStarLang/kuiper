#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "Klas_Rotg.h"

bool ok = true;

/* Klas.Rotg generates a (convention-relaxed) Givens rotation (c, s, r) from a
   pair (a, b): r = +sqrt(a*a + b*b) >= 0, c = a/r, s = b/r. We check the
   defining rotation property numerically:
       c*a + s*b == r        (rotates (a,b) onto the first axis)
       c*b - s*a == 0
   plus r == sqrt(a*a + b*b). Inputs are never both zero (r != 0). */

static bool close(double x, double y)
{
    return fabs(x - y) <= 1e-4 * (fabs(x) + fabs(y) + 1.0);
}

void test_f32(float a, float b)
{
    Klas_Rotg_rotg_out__float o = Klas_Rotg_rotg_f32(a, b);
    double c = o.rc, s = o.rs, r = o.rr;
    double er = sqrt((double)a * a + (double)b * b);
    bool this_ok = close(r, er)
        && close(c * a + s * b, r)
        && close(c * b - s * a, 0.0);
    if (!this_ok)
        ok = false;
    printf("rotg_f32(%g, %g) = (c=%g, s=%g, r=%g) %s\n", a, b, c, s, r, this_ok ? "ok" : "FAILED");
}

void test_f64(double a, double b)
{
    Klas_Rotg_rotg_out__double o = Klas_Rotg_rotg_f64(a, b);
    double c = o.rc, s = o.rs, r = o.rr;
    double er = sqrt(a * a + b * b);
    bool this_ok = close(r, er)
        && close(c * a + s * b, r)
        && close(c * b - s * a, 0.0);
    if (!this_ok)
        ok = false;
    printf("rotg_f64(%g, %g) = (c=%g, s=%g, r=%g) %s\n", a, b, c, s, r, this_ok ? "ok" : "FAILED");
}

int main()
{
    test_f32(3.0f, 4.0f);
    test_f32(4.0f, 3.0f);
    test_f32(1.0f, 0.0f);
    test_f32(0.0f, 1.0f);
    test_f32(-3.0f, 4.0f);
    test_f32(3.0f, -4.0f);
    test_f32(-3.0f, -4.0f);
    test_f32(5.0f, 12.0f);
    test_f32(1.0f, 1.0f);
    test_f32(-1.0f, 0.0f);
    test_f32(0.0f, -1.0f);

    test_f64(3.0, 4.0);
    test_f64(4.0, 3.0);
    test_f64(1.0, 0.0);
    test_f64(0.0, 1.0);
    test_f64(-3.0, 4.0);
    test_f64(3.0, -4.0);
    test_f64(-3.0, -4.0);
    test_f64(5.0, 12.0);
    test_f64(1.0, 1.0);
    test_f64(1000.0, 0.001);
    test_f64(-7.0, 24.0);

    return !ok;
}
