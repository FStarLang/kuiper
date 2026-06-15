#include <stdio.h>
#include <stdint.h>
#include "Klas_Tbmv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS tbmv (lower, non-unit): y := A*x, A n x n lower-triangular BAND with k
   sub-diagonals in cuBLAS column-major band storage (leading dim k+1): entry
   A(i,j) with j<=i<=j+k at AB[(i-j) + j*(k+1)] = AB[i + j*k]. AB length
   np = (k+1)*n. y[i] = sum_{i-k<=j<=i} AB[i+j*k]*x[j]. Exact integers. */

#define TBMV Klas_Tbmv_tbmv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, const float *AB, const float *x,
                  const float *expy)
{
    int np = (k + 1) * n;
    float *gA = dev(AB, np), *gx = dev(x, n), *gy = dev(x, n);
    float y[64];

    TBMV((uint32_t) n, (uint32_t) k, (uint32_t) np, gA, gx, gy);

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
    /* k=0 (diagonal): AB[i] = A[i][i] = [2,3,4]; x=[1,2,3]; y=[2,6,12]. */
    float AB0[3] = { 2, 3, 4 };
    float x3[3] = { 1, 2, 3 };
    float y0[3] = { 2, 6, 12 };
    check("tbmv_k0", 3, 0, AB0, x3, y0);

    /* k=1 (lower bidiagonal), n=4. AB[i+j] = A[i][j]:
       diag [2,3,4,5], sub [1,1,1]; AB=[2,1,3,1,4,1,5,G]. x=ones.
       y=[2, 1+3, 1+4, 1+5] = [2,4,5,6]. */
    float AB1[8] = { 2, 1, 3, 1, 4, 1, 5, 0 };
    float x4[4] = { 1, 1, 1, 1 };
    float y1[4] = { 2, 4, 5, 6 };
    check("tbmv_k1", 4, 1, AB1, x4, y1);

    /* k=2, n=4. AB[i+2j]=A[i][j]:
       A00=1,A10=2,A11=3,A20=4,A21=5,A22=6,A31=7,A32=8,A33=9.
       AB=[1,2,4,3,5,7,6,8,G,9,G,G]. x=ones.
       y=[1, 2+3, 4+5+6, 7+8+9] = [1,5,15,24]. */
    float AB2[12] = { 1, 2, 4, 3, 5, 7, 6, 8, 0, 9, 0, 0 };
    float y2[4] = { 1, 5, 15, 24 };
    check("tbmv_k2", 4, 2, AB2, x4, y2);

    return !ok;
}
