#include <stdio.h>
#include <stdint.h>
#include "Klas_Symm.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS symm (left, lower): X := alpha*A*B + beta*C, A n x n SYMMETRIC with
   only the lower triangle stored (row-major; A[i][j] for j>i read from
   A[j*n+i]). B,C,X are n x k column-major. Out-of-place output X. The strict
   upper triangle of A is filled with garbage to confirm it is reconstructed
   from the lower triangle. Exact small integers. */

#define SYMM Klas_Symm_symm_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, float alpha, float beta, const float *A,
                  const float *B, const float *C, const float *expX)
{
    float *gA = dev(A, n * n), *gB = dev(B, n * k), *gC = dev(C, n * k), *gX = dev(C, n * k);
    float X[64];

    SYMM((uint32_t) n, (uint32_t) k, alpha, beta, gA, gB, gC, gX);

    MUST(cudaMemcpy(X, gX, n * k * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
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
    /* A_sym=[[2,1,4],[1,3,5],[4,5,6]] (lower stored, upper garbage), n=3, k=2.
       B col0=[1,2,3], col1=[1,0,1] (column-major).
       A*B col0=[16,22,32], col1=[6,6,10]. C = ones. */
    float A3[9] = { 2, 99, 99, 1, 3, 99, 4, 5, 6 };
    float B3[6] = { 1, 2, 3, 1, 0, 1 };
    float C3[6] = { 1, 1, 1, 1, 1, 1 };

    float X1[6] = { 17, 23, 33, 7, 7, 11 };     /* alpha=1, beta=1 */
    check("symm3x2_11", 3, 2, 1.0f, 1.0f, A3, B3, C3, X1);

    float X2[6] = { 32, 44, 64, 12, 12, 20 };   /* alpha=2, beta=0 */
    check("symm3x2_20", 3, 2, 2.0f, 0.0f, A3, B3, C3, X2);

    /* n=2, k=1. A_sym=[[5,2],[2,4]], B=[3,1], A*B=[17,10], C=[100,200]. */
    float A2[4] = { 5, 99, 2, 4 };
    float B2[2] = { 3, 1 };
    float C2[2] = { 100, 200 };
    float X3[2] = { 117, 210 }; /* alpha=1, beta=1 */
    check("symm2x1_11", 2, 1, 1.0f, 1.0f, A2, B2, C2, X3);

    return !ok;
}
