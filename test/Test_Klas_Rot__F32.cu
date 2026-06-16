#include <stdio.h>
#include <stdint.h>
#include "Klas_Rot.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* c=2, s=3, x[i]=i, y[i]=2i  =>  x'=c*x+s*y=8i,  y'=c*y-s*x=i.
   All exact small integers. */
void test(int siz)
{
    float *a = (float *)malloc((siz ? siz : 1) * sizeof a[0]);
    float *b = (float *)malloc((siz ? siz : 1) * sizeof b[0]);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], siz);
    int i;
    bool this_ok = true;

    for (i = 0; i < siz; i++) {
        a[i] = (float)i;
        b[i] = 2.0f * (float)i;
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));

    Klas_Rot_rot_f32(2.0f, 3.0f, siz, ga, gb);

    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));

    for (i = 0; i < siz; i++)
        if (a[i] != 8.0f * (float)i || b[i] != (float)i)
            this_ok = false;
    if (!this_ok)
        ok = false;
    printf("test(%d) = %s\n", siz, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
}

/* complex Csrot/Zdrot: REAL c=2, s=3 applied to COMPLEX vectors.
   x[i]=(i,0), y[i]=(0,i)  =>  x'=2x+3y=(2i,3i),  y'=2y-3x=(-3i,2i). Exact. */
static void test_complex_f32(int siz)
{
    int n = siz ? siz : 1;
    cuFloatComplex *a = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *b = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *ga = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), siz);
    cuFloatComplex *gb = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), siz);
    bool this_ok = true;
    int i;

    for (i = 0; i < siz; i++) {
        a[i] = make_cuFloatComplex((float)i, 0.0f);
        b[i] = make_cuFloatComplex(0.0f, (float)i);
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_Rot_csrot_cf32(2.0f, 3.0f, siz, ga, gb);
    MUST(cudaMemcpy(a, ga, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    for (i = 0; i < siz; i++)
        if (cuCrealf(a[i]) != 2.0f * (float)i || cuCimagf(a[i]) != 3.0f * (float)i
            || cuCrealf(b[i]) != -3.0f * (float)i || cuCimagf(b[i]) != 2.0f * (float)i)
            this_ok = false;
    if (!this_ok)
        ok = false;
    printf("test_complex_f32(%d) = %s\n", siz, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
}

static void test_complex_f64(int siz)
{
    int n = siz ? siz : 1;
    cuDoubleComplex *a = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *b = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *ga = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), siz);
    cuDoubleComplex *gb = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), siz);
    bool this_ok = true;
    int i;

    for (i = 0; i < siz; i++) {
        a[i] = make_cuDoubleComplex((double)i, 0.0);
        b[i] = make_cuDoubleComplex(0.0, (double)i);
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_Rot_csrot_cf64(2.0, 3.0, siz, ga, gb);
    MUST(cudaMemcpy(a, ga, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    for (i = 0; i < siz; i++)
        if (cuCreal(a[i]) != 2.0 * (double)i || cuCimag(a[i]) != 3.0 * (double)i
            || cuCreal(b[i]) != -3.0 * (double)i || cuCimag(b[i]) != 2.0 * (double)i)
            this_ok = false;
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
    test(2049);

    test_complex_f32(1);
    test_complex_f32(513);
    test_complex_f32(1024);
    test_complex_f64(1);
    test_complex_f64(513);
    test_complex_f64(1024);
    return !ok;
}
