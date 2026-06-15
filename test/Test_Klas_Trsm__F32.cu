#include <stdio.h>
#include <stdint.h>
#include "Klas_Trsm.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS trsm (left, lower, non-unit, no transpose): solve A*X = B for k
   right-hand sides. A is n x n row-major (length n*n); B and X are n x k stored
   column-major (column c at offset c*n). Exact small-integer solutions. */

#define TRSM Klas_Trsm_trsm_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, const float *A, const float *B, const float *expX)
{
    float *gA = dev(A, n * n), *gB = dev(B, n * k), *gX = dev(B, n * k);
    float X[64];

    TRSM((uint32_t) n, (uint32_t) k, gA, gB, gX);

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
    /* A = [[2,0,0],[1,3,0],[4,5,1]] (row-major).
       col0: x=[2,3,4] => b=[4,11,27]; col1: x=[1,1,1] => b=[2,4,10]. */
    float A3[9] = { 2, 0, 0, 1, 3, 0, 4, 5, 1 };
    float B3[6] = { 4, 11, 27, /* col0 */ 2, 4, 10 /* col1 */  };
    float X3[6] = { 2, 3, 4, 1, 1, 1 };
    check("trsm_3x2", 3, 2, A3, B3, X3);

    /* k=1 is plain trsv. */
    float B1[3] = { 4, 11, 27 };
    float X1[3] = { 2, 3, 4 };
    check("trsm_3x1", 3, 1, A3, B1, X1);

    /* 2x2 system, 3 right-hand sides. A=[[5,0],[2,4]].
       cols x=[1,2],[2,1],[0,1] => b=[5,10],[10,8],[0,4]. */
    float A2[4] = { 5, 0, 2, 4 };
    float B2[6] = { 5, 10, 10, 8, 0, 4 };
    float X2[6] = { 1, 2, 2, 1, 0, 1 };
    check("trsm_2x3", 2, 3, A2, B2, X2);

    return !ok;
}
