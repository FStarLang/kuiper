#include <stdio.h>
#include <stdint.h>
#include "Klas_Geam.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* alpha=3, beta=5, A[i]=i, B[i]=2i  =>  C[i] = 3i + 10i = 13i.
   All exact small integers. */
void test(int len)
{
    float *a = (float *)malloc((len ? len : 1) * sizeof a[0]);
    float *b = (float *)malloc((len ? len : 1) * sizeof b[0]);
    float *c = (float *)malloc((len ? len : 1) * sizeof c[0]);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], len);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], len);
    float *gc = (float *)KPR_GPU_ALLOC(sizeof gc[0], len);
    int i;
    bool this_ok = true;

    for (i = 0; i < len; i++) {
        a[i] = (float)i;
        b[i] = 2.0f * (float)i;
        c[i] = -1.0f;
    }
    MUST(cudaMemcpy(ga, a, len * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, len * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gc, c, len * sizeof(float), cudaMemcpyHostToDevice));

    Klas_Geam_geam_f32(3.0f, 5.0f, len, gc, ga, gb);

    MUST(cudaMemcpy(c, gc, len * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    MUST(cudaFree(gc));

    for (i = 0; i < len; i++)
        if (c[i] != 13.0f * (float)i)
            this_ok = false;
    if (!this_ok)
        ok = false;
    printf("test(%d) = %s\n", len, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
    free(c);
}

/* complex (cuBLAS Cgeam/Zgeam): C := alpha*A + beta*B elementwise.
   alpha=(1,1), beta=(2,-1), A[i]=(i,0), B[i]=(0,i)  =>  C[i]=(2i,3i). Exact. */
static void test_complex_f32(int len)
{
    int n = len ? len : 1;
    cuFloatComplex *a = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *b = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *c = (cuFloatComplex *) malloc(n * sizeof(cuFloatComplex));
    cuFloatComplex *ga = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), len);
    cuFloatComplex *gb = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), len);
    cuFloatComplex *gc = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), len);
    cuFloatComplex alpha = make_cuFloatComplex(1.0f, 1.0f);
    cuFloatComplex beta = make_cuFloatComplex(2.0f, -1.0f);
    bool this_ok = true;
    int i;

    for (i = 0; i < len; i++) {
        a[i] = make_cuFloatComplex((float)i, 0.0f);
        b[i] = make_cuFloatComplex(0.0f, (float)i);
        c[i] = make_cuFloatComplex(-1.0f, -1.0f);
    }
    MUST(cudaMemcpy(ga, a, len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gc, c, len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_Geam_geam_cf32(alpha, beta, len, gc, ga, gb);
    MUST(cudaMemcpy(c, gc, len * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    MUST(cudaFree(gc));
    for (i = 0; i < len; i++) {
        cuFloatComplex e = cuCaddf(cuCmulf(alpha, make_cuFloatComplex((float)i, 0.0f)),
                                   cuCmulf(beta, make_cuFloatComplex(0.0f, (float)i)));
        if (cuCrealf(c[i]) != cuCrealf(e) || cuCimagf(c[i]) != cuCimagf(e))
            this_ok = false;
    }
    if (!this_ok)
        ok = false;
    printf("test_complex_f32(%d) = %s\n", len, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
    free(c);
}

static void test_complex_f64(int len)
{
    int n = len ? len : 1;
    cuDoubleComplex *a = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *b = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *c = (cuDoubleComplex *) malloc(n * sizeof(cuDoubleComplex));
    cuDoubleComplex *ga = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), len);
    cuDoubleComplex *gb = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), len);
    cuDoubleComplex *gc = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), len);
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 1.0);
    cuDoubleComplex beta = make_cuDoubleComplex(2.0, -1.0);
    bool this_ok = true;
    int i;

    for (i = 0; i < len; i++) {
        a[i] = make_cuDoubleComplex((double)i, 0.0);
        b[i] = make_cuDoubleComplex(0.0, (double)i);
        c[i] = make_cuDoubleComplex(-1.0, -1.0);
    }
    MUST(cudaMemcpy(ga, a, len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gc, c, len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_Geam_geam_cf64(alpha, beta, len, gc, ga, gb);
    MUST(cudaMemcpy(c, gc, len * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    MUST(cudaFree(gc));
    for (i = 0; i < len; i++) {
        cuDoubleComplex e = cuCadd(cuCmul(alpha, make_cuDoubleComplex((double)i, 0.0)),
                                   cuCmul(beta, make_cuDoubleComplex(0.0, (double)i)));
        if (cuCreal(c[i]) != cuCreal(e) || cuCimag(c[i]) != cuCimag(e))
            this_ok = false;
    }
    if (!this_ok)
        ok = false;
    printf("test_complex_f64(%d) = %s\n", len, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
    free(c);
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

    test_complex_f32(1);
    test_complex_f32(513);
    test_complex_f32(1024);
    test_complex_f64(1);
    test_complex_f64(513);
    test_complex_f64(1024);
    return !ok;
}
