#include <stdio.h>
#include <stdint.h>
#include "Klas_GEMM_NaiveAB.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* All values are exact small integers. Row-major layouts. */

#define GEMM Klas_GEMM_NaiveAB_gemm_f32_rrr

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

/* gemv: m=4, n=1, k=3. A[i][l]=i+l, x[l]=l, y0=10, alpha=2, beta=3.
   y[i] = 3*10 + 2*sum_l (i+l)*l = 40 + 6i. */
static void test_gemv(void)
{
    float A[12] = { 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5 };
    float x[3] = { 0, 1, 2 };
    float y[4] = { 10, 10, 10, 10 };
    float *gA = dev(A, 12), *gx = dev(x, 3), *gy = dev(y, 4);

    GEMM(4, 1, 3, 2.0f, 3.0f, gA, gx, gy);

    MUST(cudaMemcpy(y, gy, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));

    bool t = true;
    for (int i = 0; i < 4; i++)
        if (y[i] != (float)(40 + 6 * i))
            t = false;
    if (!t)
        ok = false;
    printf("test_gemv = %s\n", t ? "ok" : "FAILED");
}

/* general gemm: m=2, n=2, k=3, alpha=1, beta=0.
   A=[[1,2,3],[4,5,6]], B=[[1,0],[0,1],[1,1]] => C=[[4,5],[10,11]]. */
static void test_gemm(void)
{
    float A[6] = { 1, 2, 3, 4, 5, 6 };
    float B[6] = { 1, 0, 0, 1, 1, 1 };
    float C[4] = { 0, 0, 0, 0 };
    float exp[4] = { 4, 5, 10, 11 };
    float *gA = dev(A, 6), *gB = dev(B, 6), *gC = dev(C, 4);

    GEMM(2, 2, 3, 1.0f, 0.0f, gA, gB, gC);

    MUST(cudaMemcpy(C, gC, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));

    bool t = true;
    for (int i = 0; i < 4; i++)
        if (C[i] != exp[i])
            t = false;
    if (!t)
        ok = false;
    printf("test_gemm = %s\n", t ? "ok" : "FAILED");
}

int main()
{
    test_gemv();
    test_gemm();
    return !ok;
}
