#include <stdio.h>
#include <stdint.h>
#include "Klas_Spmv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS spmv (lower): y := alpha*A*x + beta*y, A n x n SYMMETRIC in PACKED
   row-major storage of its lower triangle (entry (i,j), j<=i, at offset
   off(i)+j; for j>i read off(j)+i). AP has length np = n*(n+1)/2. y in/out.
   Exact small integers. */

#define SPMV Klas_Spmv_spmv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, float alpha, float beta, const float *AP,
                  const float *x, const float *y0, const float *expy)
{
    int np = n * (n + 1) / 2;
    float *gA = dev(AP, np), *gx = dev(x, n), *gy = dev(y0, n);
    float y[64];

    SPMV((uint32_t) n, (uint32_t) np, alpha, beta, gA, gx, gy);

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
    /* A_sym=[[2,1,4],[1,3,5],[4,5,6]], lower packed = [2, 1,3, 4,5,6].
       x=[1,2,3]; A*x=[16,22,32]. */
    float AP3[6] = { 2, 1, 3, 4, 5, 6 };
    float x3[3] = { 1, 2, 3 };
    float y3[3] = { 10, 20, 30 };

    float e1[3] = { 26, 42, 62 };       /* alpha=1, beta=1 */
    check("spmv3_11", 3, 1.0f, 1.0f, AP3, x3, y3, e1);

    float e2[3] = { 32, 44, 64 };       /* alpha=2, beta=0 */
    check("spmv3_20", 3, 2.0f, 0.0f, AP3, x3, y3, e2);

    /* A_sym=[[5,2],[2,4]] packed [5, 2,4]; x=[3,1], y0=[1,1]; A*x=[17,10]. */
    float AP2[3] = { 5, 2, 4 };
    float x2[2] = { 3, 1 };
    float y2[2] = { 1, 1 };
    float e3[2] = { 18, 11 };   /* alpha=1, beta=1 */
    check("spmv2_11", 2, 1.0f, 1.0f, AP2, x2, y2, e3);

    return !ok;
}
