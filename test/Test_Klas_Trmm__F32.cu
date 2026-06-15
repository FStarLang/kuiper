#include <stdio.h>
#include <stdint.h>
#include "Klas_Trmm.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS trmm (left, lower, non-unit, no transpose): X := alpha*A*B. A is
   n x n row-major (length n*n); B and X are n x k stored column-major (column c
   at offset c*n). Only the lower triangle of A (j<=i) is referenced, so the
   strict upper triangle is filled with garbage to check it is ignored. Exact
   small integers. */

#define TRMM Klas_Trmm_trmm_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, float alpha, const float *A, const float *B,
                  const float *expX)
{
    float *gA = dev(A, n * n), *gB = dev(B, n * k), *gX = dev(B, n * k);
    float X[64];

    TRMM((uint32_t) n, (uint32_t) k, alpha, gA, gB, gX);

    MUST(cudaMemcpy(X, gX, n * k * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gX));

    bool t = true;
    for (int i = 0; i < n * k; i++)
        if (X[i] != expX[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n * k; i++)
        printf(" %g", X[i]);
    printf("\n");
}

int main()
{
    /* A lower-tri [[2,.,.],[1,3,.],[4,5,1]] (upper=garbage 99), n=3, k=2.
       B col0=[1,2,3], col1=[1,0,1] (column-major).
       X[c][i] = alpha * sum_{j<=i} A[i][j]*B[c][j].
       alpha=1: col0=[2, 1+6, 4+10+3]=[2,7,17]; col1=[2, 1, 4+0+1]=[2,1,5]. */
    float A3[9] = { 2, 99, 99, 1, 3, 99, 4, 5, 1 };
    float B3[6] = { 1, 2, 3, 1, 0, 1 };
    float X3[6] = { 2, 7, 17, 2, 1, 5 };
    check("trmm_3x2_a1", 3, 2, 1.0f, A3, B3, X3);

    /* same with alpha=2 => all doubled. */
    float X3b[6] = { 4, 14, 34, 4, 2, 10 };
    check("trmm_3x2_a2", 3, 2, 2.0f, A3, B3, X3b);

    /* k=1 is plain trmv (alpha=1). */
    float B1[3] = { 1, 2, 3 };
    float X1[3] = { 2, 7, 17 };
    check("trmm_3x1", 3, 1, 1.0f, A3, B1, X1);

    /* 2x2 A=[[5,.],[2,4]] (upper garbage), k=2, alpha=3.
       B col0=[3,1], col1=[1,2].
       col0=[5*3, 2*3+4*1]=[15,10]; col1=[5*1, 2*1+4*2]=[5,10]; *3. */
    float A2[4] = { 5, 77, 2, 4 };
    float B2[4] = { 3, 1, 1, 2 };
    float X2[4] = { 45, 30, 15, 30 };
    check("trmm_2x2_a3", 2, 2, 3.0f, A2, B2, X2);

    return !ok;
}
