#include <stdio.h>
#include <stdint.h>
#include "Klas_RowScale.h"

bool ok = true;

/* Klas.RowScale = cuBLAS dgmm in LEFT mode (in-place): B := diag(a) * B, i.e.
   row i of the m x n matrix B is scaled by a[i]. Two storage layouts:
     row-major: element (i,j) at b[i*n + j]
     col-major: element (i,j) at b[j*m + i]
   All values are small exact integers. */

#define M 3
#define N 4

static void report(const char *name, bool this_ok)
{
    if (!this_ok)
        ok = false;
    printf("%s = %s\n", name, this_ok ? "ok" : "FAILED");
}

static void test_f32_rowmajor(void)
{
    float a[M], b[M * N];
    int i, j;
    for (i = 0; i < M; i++)
        a[i] = (float)(i + 1);
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            b[i * N + j] = (float)(i * 10 + j);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), M);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof(float), M * N);
    MUST(cudaMemcpy(ga, a, M * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, M * N * sizeof(float), cudaMemcpyHostToDevice));
    Klas_RowScale_rowscale_f32_rowmajor(M, N, ga, gb);
    MUST(cudaMemcpy(b, gb, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    bool t = true;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            if (b[i * N + j] != (float)(i + 1) * (float)(i * 10 + j))
                t = false;
    report("test_f32_rowmajor", t);
}

static void test_cf32_rowmajor(void)
{
    cuFloatComplex a[M], b[M * N];
    int i, j;
    for (i = 0; i < M; i++)
        a[i] = make_cuFloatComplex((float)(i + 1), 1.0f);
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            b[i * N + j] = make_cuFloatComplex((float)(i * 10 + j), 0.0f);
    cuFloatComplex *ga = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), M);
    cuFloatComplex *gb = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), M * N);
    MUST(cudaMemcpy(ga, a, M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, M * N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_RowScale_rowscale_cf32_rowmajor(M, N, ga, gb);
    MUST(cudaMemcpy(b, gb, M * N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    bool t = true;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            cuFloatComplex e = cuCmulf(make_cuFloatComplex((float)(i + 1), 1.0f),
                                       make_cuFloatComplex((float)(i * 10 + j), 0.0f));
            cuFloatComplex got = b[i * N + j];
            if (cuCrealf(got) != cuCrealf(e) || cuCimagf(got) != cuCimagf(e))
                t = false;
        }
    report("test_cf32_rowmajor", t);
}

static void test_cf32_colmajor(void)
{
    cuFloatComplex a[M], b[M * N];
    int i, j;
    for (i = 0; i < M; i++)
        a[i] = make_cuFloatComplex((float)(i + 1), 1.0f);
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            b[j * M + i] = make_cuFloatComplex((float)(i * 10 + j), 0.0f);
    cuFloatComplex *ga = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), M);
    cuFloatComplex *gb = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), M * N);
    MUST(cudaMemcpy(ga, a, M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, M * N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    Klas_RowScale_rowscale_cf32_colmajor(M, N, ga, gb);
    MUST(cudaMemcpy(b, gb, M * N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    bool t = true;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            cuFloatComplex e = cuCmulf(make_cuFloatComplex((float)(i + 1), 1.0f),
                                       make_cuFloatComplex((float)(i * 10 + j), 0.0f));
            cuFloatComplex got = b[j * M + i];
            if (cuCrealf(got) != cuCrealf(e) || cuCimagf(got) != cuCimagf(e))
                t = false;
        }
    report("test_cf32_colmajor", t);
}

static void test_cf64_rowmajor(void)
{
    cuDoubleComplex a[M], b[M * N];
    int i, j;
    for (i = 0; i < M; i++)
        a[i] = make_cuDoubleComplex((double)(i + 1), 1.0);
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            b[i * N + j] = make_cuDoubleComplex((double)(i * 10 + j), 0.0);
    cuDoubleComplex *ga = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), M);
    cuDoubleComplex *gb = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), M * N);
    MUST(cudaMemcpy(ga, a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    Klas_RowScale_rowscale_cf64_rowmajor(M, N, ga, gb);
    MUST(cudaMemcpy(b, gb, M * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    bool t = true;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            cuDoubleComplex e = cuCmul(make_cuDoubleComplex((double)(i + 1), 1.0),
                                       make_cuDoubleComplex((double)(i * 10 + j), 0.0));
            cuDoubleComplex got = b[i * N + j];
            if (cuCreal(got) != cuCreal(e) || cuCimag(got) != cuCimag(e))
                t = false;
        }
    report("test_cf64_rowmajor", t);
}

int main()
{
    test_f32_rowmajor();
    test_cf32_rowmajor();
    test_cf32_colmajor();
    test_cf64_rowmajor();
    return !ok;
}
