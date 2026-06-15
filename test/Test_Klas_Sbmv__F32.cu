#include <stdio.h>
#include <stdint.h>
#include "Klas_Sbmv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS sbmv (lower): y := alpha*A*x + beta*y, A n x n SYMMETRIC BAND with k
   off-diagonals, lower band stored in cuBLAS column-major band layout (A(i,j),
   j<=i, at AB[i+j*k]; for j>i read AB[j+i*k]). AB length np=(k+1)*n. y in/out.
   Exact small integers. */

#define SBMV Klas_Sbmv_sbmv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, float alpha, float beta, const float *AB,
                  const float *x, const float *y0, const float *expy)
{
    int np = (k + 1) * n;
    float *gA = dev(AB, np), *gx = dev(x, n), *gy = dev(y0, n);
    float y[64];

    SBMV((uint32_t) n, (uint32_t) k, (uint32_t) np, alpha, beta, gA, gx, gy);

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
    /* k=0 (diagonal): AB=[2,3,4]; x=[1,2,3]; A*x=[2,6,12]. */
    float AB0[3] = { 2, 3, 4 };
    float x3[3] = { 1, 2, 3 };
    float z3[3] = { 0, 0, 0 };
    float e0[3] = { 2, 6, 12 };
    check("sbmv_k0", 3, 0, 1.0f, 0.0f, AB0, x3, z3, e0);

    /* k=1 (symmetric tridiagonal), n=4. diag [2,3,4,5], off [1,1,1];
       AB=[2,1,3,1,4,1,5,G]. x=ones. A*x=[3,5,6,6]. */
    float AB1[8] = { 2, 1, 3, 1, 4, 1, 5, 0 };
    float x4[4] = { 1, 1, 1, 1 };
    float y4[4] = { 10, 20, 30, 40 };

    float e1[4] = { 13, 25, 36, 46 };   /* alpha=1, beta=1 */
    check("sbmv_k1_11", 4, 1, 1.0f, 1.0f, AB1, x4, y4, e1);

    float e2[4] = { 6, 10, 12, 12 };    /* alpha=2, beta=0 */
    float z4[4] = { 0, 0, 0, 0 };
    check("sbmv_k1_20", 4, 1, 2.0f, 0.0f, AB1, x4, z4, e2);

    /* k=2 (symmetric pentadiagonal), n=4. AB[i+2j]=A(i,j) lower:
       A00=1,A10=1,A20=1,A11=2,A21=1,A31=1,A22=3,A32=1,A33=4.
       AB=[1,1,1,2,1,1,3,1,G,4,G,G]. x=ones. A*x=[3,5,6,6]. */
    float AB2[12] = { 1, 1, 1, 2, 1, 1, 3, 1, 0, 4, 0, 0 };
    float e3[4] = { 3, 5, 6, 6 };       /* alpha=1, beta=0 */
    check("sbmv_k2_10", 4, 2, 1.0f, 0.0f, AB2, x4, z4, e3);

    return !ok;
}
