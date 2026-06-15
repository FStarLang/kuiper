#include <stdio.h>
#include <stdint.h>
#include "Klas_Trmv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS trmv (lower, non-unit): y := A*x with y[i] = sum_{j<=i} A[i][j]*x[j].
   A is n x n row-major; the strict upper triangle must be ignored (we fill it
   with garbage to check). Exact small integers. */

#define TRMV Klas_Trmv_trmv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, const float *A, const float *x, const float *expy)
{
    float *gA = dev(A, n * n), *gx = dev(x, n), *gy = dev(x, n);
    float y[64];

    TRMV((uint32_t) n, gA, gx, gy);

    MUST(cudaMemcpy(y, gy, n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));

    bool t = true;
    for (int i = 0; i < n; i++)
        if (y[i] != expy[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n; i++)
        printf(" %g", y[i]);
    printf("\n");
}

int main()
{
    /* lower triangle [[2,.,.],[1,3,.],[4,5,1]], x=[1,2,3]; upper = garbage (99).
       y = [2, 1+6, 4+10+3] = [2, 7, 17]. */
    float A3[9] = { 2, 99, 99, 1, 3, 99, 4, 5, 1 };
    float x3[3] = { 1, 2, 3 };
    float y3[3] = { 2, 7, 17 };
    check("trmv3", 3, A3, x3, y3);

    /* 2x2 [[5,.],[2,4]], x=[3,1]; y=[15, 6+4]=[15,10]. */
    float A2[4] = { 5, 77, 2, 4 };
    float x2[2] = { 3, 1 };
    float y2[2] = { 15, 10 };
    check("trmv2", 2, A2, x2, y2);

    return !ok;
}
