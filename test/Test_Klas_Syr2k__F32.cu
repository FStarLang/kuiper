#include <stdio.h>
#include <stdint.h>
#include "Klas_Syr2k.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS syr2k (lower, trans=N): X := alpha*(A*B^T + B*A^T) + beta*C. A,B are
   n x k row-major (length n*k); C,X are n x n row-major (the result is
   symmetric). Out-of-place output X. Exact small integers. */

#define SYR2K Klas_Syr2k_syr2k_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, float alpha, float beta, const float *A,
                  const float *B, const float *C, const float *expX)
{
    float *gA = dev(A, n * k), *gB = dev(B, n * k), *gC = dev(C, n * n), *gX = dev(C, n * n);
    float X[64];

    SYR2K((uint32_t) n, (uint32_t) k, alpha, beta, gA, gB, gC, gX);

    MUST(cudaMemcpy(X, gX, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gB));
    MUST(cudaFree(gC));
    MUST(cudaFree(gX));

    bool t = true;
    for (int i = 0; i < n * n; i++)
        if (X[i] != expX[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n * n; i++)
        printf(" %g", X[i]);
    printf("\n");
}

int main()
{
    /* n=2, k=2. A=[[1,2],[3,4]], B=[[1,0],[0,1]].
       A*B^T=[[1,2],[3,4]], B*A^T=[[1,3],[2,4]], sum=[[2,5],[5,8]]. C=ones. */
    float A1[4] = { 1, 2, 3, 4 };
    float B1[4] = { 1, 0, 0, 1 };
    float C1[4] = { 1, 1, 1, 1 };
    float X1[4] = { 3, 6, 6, 9 };       /* alpha=1, beta=1 */
    check("syr2k2_11", 2, 2, 1.0f, 1.0f, A1, B1, C1, X1);

    float X1b[4] = { 4, 10, 10, 16 };   /* alpha=2, beta=0 */
    check("syr2k2_20", 2, 2, 2.0f, 0.0f, A1, B1, C1, X1b);

    /* n=2, k=1. A=[[1],[2]], B=[[3],[4]]. sum = [[6,10],[10,16]]. */
    float A2[2] = { 1, 2 };
    float B2[2] = { 3, 4 };
    float C2[4] = { 0, 0, 0, 0 };
    float X2[4] = { 6, 10, 10, 16 };    /* alpha=1, beta=0 */
    check("syr2k2_10", 2, 1, 1.0f, 0.0f, A2, B2, C2, X2);

    return !ok;
}
