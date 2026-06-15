#include <stdio.h>
#include <stdint.h>
#include "Klas_Tpmv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS tpmv (lower, non-unit): y := A*x, A n x n lower-triangular in PACKED
   row-major form: row 0 has 1 entry, row 1 has 2, ..., so entry (i,j) (j<=i) is
   at offset i*(i+1)/2 + j. AP has length np = n*(n+1)/2.
   y[i] = sum_{j<=i} AP[off(i)+j] * x[j]. Exact small integers. */

#define TPMV Klas_Tpmv_tpmv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, const float *AP, const float *x, const float *expy)
{
    int np = n * (n + 1) / 2;
    float *gA = dev(AP, np), *gx = dev(x, n), *gy = dev(x, n);
    float y[64];

    TPMV((uint32_t) n, (uint32_t) np, gA, gx, gy);

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
    /* lower-tri [[2,.,.],[1,3,.],[4,5,6]] packed row-major = [2, 1,3, 4,5,6].
       x=[1,2,3]; y = [2, 1+6, 4+10+18] = [2, 7, 32]. */
    float AP3[6] = { 2, 1, 3, 4, 5, 6 };
    float x3[3] = { 1, 2, 3 };
    float y3[3] = { 2, 7, 32 };
    check("tpmv3", 3, AP3, x3, y3);

    /* 2x2 [[5,.],[2,4]] packed = [5, 2,4]; x=[3,1]; y=[15, 6+4]=[15,10]. */
    float AP2[3] = { 5, 2, 4 };
    float x2[2] = { 3, 1 };
    float y2[2] = { 15, 10 };
    check("tpmv2", 2, AP2, x2, y2);

    /* 4x4 lower-tri, packed = rows [1],[2,3],[4,5,6],[7,8,9,10]; x=[1,1,1,1].
       y=[1, 2+3, 4+5+6, 7+8+9+10] = [1,5,15,34]. */
    float AP4[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    float x4[4] = { 1, 1, 1, 1 };
    float y4[4] = { 1, 5, 15, 34 };
    check("tpmv4", 4, AP4, x4, y4);

    return !ok;
}
