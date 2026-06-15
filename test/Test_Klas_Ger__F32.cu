#include <stdio.h>
#include <stdint.h>
#include "Klas_Ger.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS ger:  A := alpha * x * y^T + A.  Row-major A (m x n), x (m), y (n).
   All values are exact small integers. */

#define GER Klas_Ger_ger_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

/* m=2, n=3, alpha=2, x=[1,2], y=[10,20,30], A0 = all ones.
   A[i][j] = 1 + 2*x[i]*y[j]. */
static void test_basic(void)
{
    float x[2] = { 1, 2 };
    float y[3] = { 10, 20, 30 };
    float A[6] = { 1, 1, 1, 1, 1, 1 };
    float exp[6] = { 21, 41, 61, 41, 81, 121 };
    float *gx = dev(x, 2), *gy = dev(y, 3), *gA = dev(A, 6);

    GER(2, 3, 2.0f, gx, gy, gA);

    MUST(cudaMemcpy(A, gA, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));
    MUST(cudaFree(gA));

    bool t = true;
    for (int i = 0; i < 6; i++)
        if (A[i] != exp[i])
            t = false;
    if (!t)
        ok = false;
    printf("test_basic = %s\n", t ? "ok" : "FAILED");
}

/* Accumulation: run ger twice; the second call must add on top of the first.
   m=3, n=2, alpha=1, x=[1,2,3], y=[1,10], A0 = 0.  After two calls A=2*x*y^T. */
static void test_accumulate(void)
{
    float x[3] = { 1, 2, 3 };
    float y[2] = { 1, 10 };
    float A[6] = { 0, 0, 0, 0, 0, 0 };
    float exp[6] = { 2, 20, 4, 40, 6, 60 };     /* 2 * x[i] * y[j] */
    float *gx = dev(x, 3), *gy = dev(y, 2), *gA = dev(A, 6);

    GER(3, 2, 1.0f, gx, gy, gA);
    GER(3, 2, 1.0f, gx, gy, gA);

    MUST(cudaMemcpy(A, gA, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));
    MUST(cudaFree(gA));

    bool t = true;
    for (int i = 0; i < 6; i++)
        if (A[i] != exp[i])
            t = false;
    if (!t)
        ok = false;
    printf("test_accumulate = %s\n", t ? "ok" : "FAILED");
}

int main()
{
    test_basic();
    test_accumulate();
    return !ok;
}
