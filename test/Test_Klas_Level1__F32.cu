#include <stdio.h>
#include <stdint.h>
#include "Klas_Level1.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* All values are exact small integers, so float comparisons are exact. */
void test(int siz)
{
    float *a = (float *)malloc((siz ? siz : 1) * sizeof a[0]);
    float *b = (float *)malloc((siz ? siz : 1) * sizeof b[0]);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], siz);
    int i;
    bool this_ok = true;

    /* scal: a := 3 * a, with a[i] = i  =>  a[i] = 3i */
    for (i = 0; i < siz; i++)
        a[i] = (float)i;
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_scal_f32(3.0f, siz, ga);
    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (a[i] != 3.0f * (float)i)
            this_ok = false;

    /* axpy: y := 3 * x + y, with x[i] = i, y[i] = 2i  =>  y[i] = 5i */
    for (i = 0; i < siz; i++) {
        a[i] = (float)i;        /* x */
        b[i] = 2.0f * (float)i; /* y */
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_axpy_f32(3.0f, siz, gb, ga);    /* y := 3*x + y */
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (b[i] != 5.0f * (float)i)
            this_ok = false;

    /* copy: y := x, with x[i] = i + 7 */
    for (i = 0; i < siz; i++) {
        a[i] = (float)(i + 7);  /* x */
        b[i] = -1.0f;           /* y */
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_copy_f32(siz, gb, ga);  /* y := x */
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (b[i] != (float)(i + 7))
            this_ok = false;

    /* swap: x <-> y, with x[i] = i, y[i] = -i  =>  x[i] = -i, y[i] = i */
    for (i = 0; i < siz; i++) {
        a[i] = (float)i;        /* x */
        b[i] = -(float)i;       /* y */
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_swap_f32(siz, ga, gb);
    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (a[i] != -(float)i || b[i] != (float)i)
            this_ok = false;

    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    if (!this_ok)
        ok = false;
    printf("test(%d) = %s\n", siz, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
}

/* ---- complex (cuFloatComplex): cuBLAS Cscal/Ccopy/Cswap ----
   All real/imag parts are small exact integers, so equality is exact. */
static void test_complex_f32(int siz)
{
    int n = siz ? siz : 1;
    cuFloatComplex *a = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *b = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *ga = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), siz);
    cuFloatComplex *gb = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), siz);
    cuFloatComplex alpha = make_cuFloatComplex(2.0f, 1.0f);
    bool this_ok = true;
    int i;

    /* scal: a := alpha * a, a[i] = (i, 1) */
    for (i = 0; i < siz; i++)
        a[i] = make_cuFloatComplex((float)i, 1.0f);
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_Level1_scal_cf32(alpha, siz, ga);
    MUST(cudaMemcpy(b, ga, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++) {
        cuFloatComplex e = cuCmulf(alpha, make_cuFloatComplex((float)i, 1.0f));
        if (cuCrealf(b[i]) != cuCrealf(e) || cuCimagf(b[i]) != cuCimagf(e))
            this_ok = false;
    }

    /* copy: y := x, x[i] = (i, -i) */
    for (i = 0; i < siz; i++) {
        a[i] = make_cuFloatComplex((float)i, -(float)i);
        b[i] = make_cuFloatComplex(-1.0f, -1.0f);
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_Level1_copy_cf32(siz, gb, ga);
    MUST(cudaMemcpy(b, gb, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (cuCrealf(b[i]) != (float)i || cuCimagf(b[i]) != -(float)i)
            this_ok = false;

    /* swap: x <-> y, x[i] = (i, 0), y[i] = (0, i) */
    for (i = 0; i < siz; i++) {
        a[i] = make_cuFloatComplex((float)i, 0.0f);
        b[i] = make_cuFloatComplex(0.0f, (float)i);
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_Level1_swap_cf32(siz, ga, gb);
    MUST(cudaMemcpy(a, ga, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (cuCrealf(a[i]) != 0.0f || cuCimagf(a[i]) != (float)i
            || cuCrealf(b[i]) != (float)i || cuCimagf(b[i]) != 0.0f)
            this_ok = false;

    /* csscal: a := alpha*a, REAL alpha=3, a[i]=(i, 2i) => (3i, 6i) */
    for (i = 0; i < siz; i++)
        a[i] = make_cuFloatComplex((float)i, 2.0f * (float)i);
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_Level1_csscal(3.0f, siz, ga);
    MUST(cudaMemcpy(a, ga, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (cuCrealf(a[i]) != 3.0f * (float)i || cuCimagf(a[i]) != 6.0f * (float)i)
            this_ok = false;

    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    if (!this_ok)
        ok = false;
    printf("test_complex_f32(%d) = %s\n", siz, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
}

/* ---- complex (cuDoubleComplex): cuBLAS Zscal/Zcopy/Zswap ---- */
static void test_complex_f64(int siz)
{
    int n = siz ? siz : 1;
    cuDoubleComplex *a = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *b = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *ga = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), siz);
    cuDoubleComplex *gb = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), siz);
    cuDoubleComplex alpha = make_cuDoubleComplex(2.0, 1.0);
    bool this_ok = true;
    int i;

    for (i = 0; i < siz; i++)
        a[i] = make_cuDoubleComplex((double)i, 1.0);
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_Level1_scal_cf64(alpha, siz, ga);
    MUST(cudaMemcpy(b, ga, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++) {
        cuDoubleComplex e = cuCmul(alpha, make_cuDoubleComplex((double)i, 1.0));
        if (cuCreal(b[i]) != cuCreal(e) || cuCimag(b[i]) != cuCimag(e))
            this_ok = false;
    }

    for (i = 0; i < siz; i++) {
        a[i] = make_cuDoubleComplex((double)i, -(double)i);
        b[i] = make_cuDoubleComplex(-1.0, -1.0);
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_Level1_copy_cf64(siz, gb, ga);
    MUST(cudaMemcpy(b, gb, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (cuCreal(b[i]) != (double)i || cuCimag(b[i]) != -(double)i)
            this_ok = false;

    for (i = 0; i < siz; i++) {
        a[i] = make_cuDoubleComplex((double)i, 0.0);
        b[i] = make_cuDoubleComplex(0.0, (double)i);
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_Level1_swap_cf64(siz, ga, gb);
    MUST(cudaMemcpy(a, ga, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (cuCreal(a[i]) != 0.0 || cuCimag(a[i]) != (double)i
            || cuCreal(b[i]) != (double)i || cuCimag(b[i]) != 0.0)
            this_ok = false;

    /* zdscal: a := alpha*a, REAL alpha=3, a[i]=(i, 2i) => (3i, 6i) */
    for (i = 0; i < siz; i++)
        a[i] = make_cuDoubleComplex((double)i, 2.0 * (double)i);
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_Level1_zdscal(3.0, siz, ga);
    MUST(cudaMemcpy(a, ga, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (cuCreal(a[i]) != 3.0 * (double)i || cuCimag(a[i]) != 6.0 * (double)i)
            this_ok = false;

    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    if (!this_ok)
        ok = false;
    printf("test_complex_f64(%d) = %s\n", siz, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
}

int main()
{
    test(0);
    test(1);
    test(2);
    test(511);
    test(512);
    test(513);
    test(1024);
    test(1025);
    test(2048);
    test(100000);

    test_complex_f32(1);
    test_complex_f32(2);
    test_complex_f32(513);
    test_complex_f32(1024);
    test_complex_f64(1);
    test_complex_f64(2);
    test_complex_f64(513);
    test_complex_f64(1024);
    return !ok;
}
