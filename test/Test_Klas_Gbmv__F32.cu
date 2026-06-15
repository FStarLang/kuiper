#include <stdio.h>
#include <stdint.h>
#include "Klas_Gbmv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS gbmv (no transpose, square m=n): y := alpha*A*x + beta*y, A n x n
   GENERAL BAND with kl sub- and ku super-diagonals in the cuBLAS column-major
   band layout (A(i,j), i-kl<=j<=i+ku, at AB[ku+i+j*(kl+ku)]). AB length
   np=(kl+ku+1)*n. y in/out. Exact small integers. */

#define GBMV Klas_Gbmv_gbmv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int kl, int ku, float alpha, float beta,
                  const float *AB, const float *x, const float *y0, const float *expy)
{
    int np = (kl + ku + 1) * n;
    float *gA = dev(AB, np), *gx = dev(x, n), *gy = dev(y0, n);
    float y[64];

    GBMV((uint32_t) n, (uint32_t) kl, (uint32_t) ku, (uint32_t) np, alpha, beta, gA, gx, gy);

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
    /* kl=ku=1 (general tridiagonal), n=4. A(i,j) at AB[1+i+2j]:
       diag 1,2,3,4; super A01=5,A12=6,A23=7; sub A10=8,A21=9,A32=10.
       AB=[G,1,8,5,2,9,6,3,10,7,4,G]. x=ones.
       y=[1+5, 8+2+6, 9+3+7, 10+4]=[6,16,19,14]. */
    float AB1[12] = { 0, 1, 8, 5, 2, 9, 6, 3, 10, 7, 4, 0 };
    float x4[4] = { 1, 1, 1, 1 };
    float y4[4] = { 100, 200, 300, 400 };

    float e1[4] = { 106, 216, 319, 414 };       /* alpha=1, beta=1 */
    check("gbmv_11_11", 4, 1, 1, 1.0f, 1.0f, AB1, x4, y4, e1);

    float z4[4] = { 0, 0, 0, 0 };
    float e2[4] = { 12, 32, 38, 28 };   /* alpha=2, beta=0 */
    check("gbmv_11_20", 4, 1, 1, 2.0f, 0.0f, AB1, x4, z4, e2);

    /* kl=0, ku=1 (upper bidiagonal), n=3. A(i,j) at AB[1+i+j]:
       diag 1,2,3; super A01=4,A12=5. AB=[G,1,4,2,5,3]. x=ones.
       y=[1+4, 2+5, 3]=[5,7,3]. */
    float AB2[6] = { 0, 1, 4, 2, 5, 3 };
    float x3[3] = { 1, 1, 1 };
    float z3[3] = { 0, 0, 0 };
    float e3[3] = { 5, 7, 3 };  /* alpha=1, beta=0 */
    check("gbmv_01_10", 3, 0, 1, 1.0f, 0.0f, AB2, x3, z3, e3);

    return !ok;
}
