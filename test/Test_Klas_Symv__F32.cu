#include <stdio.h>
#include <stdint.h>
#include "Klas_Symv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS symv (lower): y := alpha*A*x + beta*y, A n x n SYMMETRIC with only the
   lower triangle stored (row-major); A[i][j] for j>i is read from A[j*n+i]. The
   strict upper triangle is filled with garbage to confirm it is reconstructed
   from the lower triangle. y is in/out. Exact small integers. */

#define SYMV Klas_Symv_symv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, float alpha, float beta, const float *A,
                  const float *x, const float *y0, const float *expy)
{
    float *gA = dev(A, n * n), *gx = dev(x, n), *gy = dev(y0, n);
    float y[64];

    SYMV((uint32_t) n, alpha, beta, gA, gx, gy);

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
    /* A symmetric, lower stored [[2,.,.],[1,3,.],[4,5,6]] (upper=garbage 99);
       A_sym = [[2,1,4],[1,3,5],[4,5,6]]. x=[1,2,3].
       A*x = [2+2+12, 1+6+15, 4+10+18] = [16,22,32]. */
    float A3[9] = { 2, 99, 99, 1, 3, 99, 4, 5, 6 };
    float x3[3] = { 1, 2, 3 };
    float y3[3] = { 10, 20, 30 };

    float e1[3] = { 26, 42, 62 };       /* alpha=1, beta=1 */
    check("symv3_11", 3, 1.0f, 1.0f, A3, x3, y3, e1);

    float e2[3] = { 32, 44, 64 };       /* alpha=2, beta=0 */
    check("symv3_20", 3, 2.0f, 0.0f, A3, x3, y3, e2);

    float e3[3] = { 36, 62, 92 };       /* alpha=1, beta=2 */
    check("symv3_12", 3, 1.0f, 2.0f, A3, x3, y3, e3);

    /* 2x2 A_sym=[[5,2],[2,4]] (upper garbage), x=[3,1], y0=[1,1], a=1,b=1.
       A*x=[17,10]; y=[18,11]. */
    float A2[4] = { 5, 99, 2, 4 };
    float x2[2] = { 3, 1 };
    float y2[2] = { 1, 1 };
    float e4[2] = { 18, 11 };
    check("symv2_11", 2, 1.0f, 1.0f, A2, x2, y2, e4);

    return !ok;
}
