#include <stdio.h>
#include <stdint.h>
#include "Klas_Syr2.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS syr2 (rank-2 symmetric update): C := A + alpha*(x*y^T + y*x^T), with
   C[i][j] = A[i][j] + alpha*(x[i]*y[j] + y[i]*x[j]).  A,C are n x n row-major;
   here we use a separate output C and update the FULL matrix (both triangles).
   Exact small integers. */

#define SYR2 Klas_Syr2_syr2_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, float alpha, const float *A, const float *x,
                  const float *y, const float *expC)
{
    float *gA = dev(A, n * n), *gx = dev(x, n), *gy = dev(y, n), *gC = dev(A, n * n);
    float C[64];

    SYR2((uint32_t) n, alpha, gA, gx, gy, gC);

    MUST(cudaMemcpy(C, gC, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));
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
    /* n=2, A=[[1,2],[3,4]], x=[1,2], y=[3,4], alpha=1.
       C[i][j] = A[i][j] + (x[i]*y[j] + y[i]*x[j]):
       [[1+6, 2+10],[3+10, 4+16]] = [[7,12],[13,20]]. */
    float A2[4] = { 1, 2, 3, 4 };
    float x2[2] = { 1, 2 };
    float y2[2] = { 3, 4 };
    float C2[4] = { 7, 12, 13, 20 };
    check("syr2_2", 2, 1.0f, A2, x2, y2, C2);

    /* n=3, A=I, x=[1,0,1], y=[0,1,1], alpha=1.
       x*y^T + y*x^T = [[0,1,1],[1,0,1],[1,1,2]]; +I => [[1,1,1],[1,1,1],[1,1,3]]. */
    float A3[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    float x3[3] = { 1, 0, 1 };
    float y3[3] = { 0, 1, 1 };
    float C3[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 3 };
    check("syr2_3", 3, 1.0f, A3, x3, y3, C3);

    /* n=2, alpha=0 leaves A unchanged. */
    float A2b[4] = { 5, 6, 7, 8 };
    float C2b[4] = { 5, 6, 7, 8 };
    check("syr2_0", 2, 0.0f, A2b, x2, y2, C2b);

    return !ok;
}
