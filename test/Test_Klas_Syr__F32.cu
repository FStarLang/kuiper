#include <stdio.h>
#include <stdint.h>
#include "Klas_Syr.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS syr (rank-1 symmetric update): C := A + alpha*x*x^T, with
   C[i][j] = A[i][j] + alpha*x[i]*x[j].  A,C are n x n row-major; here we use
   a separate output C (out-of-place) and update the FULL matrix (both
   triangles).  Exact small integers. */

#define SYR Klas_Syr_syr_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, float alpha, const float *A, const float *x,
                  const float *expC)
{
    float *gA = dev(A, n * n), *gx = dev(x, n), *gC = dev(A, n * n);
    float C[64];

    SYR((uint32_t) n, alpha, gA, gx, gC);

    MUST(cudaMemcpy(C, gC, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gx));
    MUST(cudaFree(gC));

    bool t = true;
    for (int i = 0; i < n * n; i++)
        if (C[i] != expC[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n * n; i++)
        printf(" %g", C[i]);
    printf("\n");
}

int main()
{
    /* n=2, A=[[1,2],[3,4]], x=[1,2], alpha=2.
       C[i][j] = A[i][j] + 2*x[i]*x[j]:
       [[1+2, 2+4],[3+4, 4+8]] = [[3,6],[7,12]]. */
    float A2[4] = { 1, 2, 3, 4 };
    float x2[2] = { 1, 2 };
    float C2[4] = { 3, 6, 7, 12 };
    check("syr2", 2, 2.0f, A2, x2, C2);

    /* n=3, A=I, x=[1,2,3], alpha=1.  C[i][j] = I[i][j] + x[i]*x[j]:
       [[2,2,3],[2,5,6],[3,6,10]]. */
    float A3[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    float x3[3] = { 1, 2, 3 };
    float C3[9] = { 2, 2, 3, 2, 5, 6, 3, 6, 10 };
    check("syr3", 3, 1.0f, A3, x3, C3);

    /* n=3, alpha=0 leaves A unchanged. */
    float A3b[9] = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    float C3b[9] = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    check("syr0", 3, 0.0f, A3b, x3, C3b);

    return !ok;
}
