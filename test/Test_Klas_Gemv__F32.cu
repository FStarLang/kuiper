#include <stdio.h>
#include <stdint.h>
#include "Klas_Gemv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS gemv (no transpose):  y := alpha * A * x + beta * y.
   A row-major m x k, x is k-vector, y is m-vector. Exact small integers. */

#define GEMV Klas_Gemv_gemv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

/* m=3, k=2, alpha=2, beta=3, A=[[1,2],[3,4],[5,6]], x=[1,1], y0=[10,10,10].
   A*x = [3,7,11]; y = 3*10 + 2*(A*x) = [36,44,52]. */
static void test_basic(void)
{
    float A[6] = { 1, 2, 3, 4, 5, 6 };
    float x[2] = { 1, 1 };
    float y[3] = { 10, 10, 10 };
    float exp[3] = { 36, 44, 52 };
    float *gA = dev(A, 6), *gx = dev(x, 2), *gy = dev(y, 3);

    GEMV(3, 2, 2.0f, 3.0f, gA, gx, gy);

    MUST(cudaMemcpy(y, gy, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));

    bool t = true;
    for (int i = 0; i < 3; i++)
        if (y[i] != exp[i])
            t = false;
    if (!t)
        ok = false;
    printf("test_basic = %s\n", t ? "ok" : "FAILED");
}

/* beta = 0 overwrites: y := A*x.  m=2,k=3, A=[[1,0,2],[0,3,0]], x=[5,6,7].
   y = [5+14, 18] = [19,18]. */
static void test_beta0(void)
{
    float A[6] = { 1, 0, 2, 0, 3, 0 };
    float x[3] = { 5, 6, 7 };
    float y[2] = { 99, 99 };
    float exp[2] = { 19, 18 };
    float *gA = dev(A, 6), *gx = dev(x, 3), *gy = dev(y, 2);

    GEMV(2, 3, 1.0f, 0.0f, gA, gx, gy);

    MUST(cudaMemcpy(y, gy, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));

    bool t = true;
    for (int i = 0; i < 2; i++)
        if (y[i] != exp[i])
            t = false;
    if (!t)
        ok = false;
    printf("test_beta0 = %s\n", t ? "ok" : "FAILED");
}

int main()
{
    test_basic();
    test_beta0();
    return !ok;
}
