#include <stdio.h>
#include <stdint.h>
#include "Klas_Tbsv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS tbsv (lower, non-unit): solve A*y=b, A n x n lower-triangular BAND with
   k sub-diagonals in cuBLAS column-major band storage (A(i,j) at AB[i+j*k]; AB
   length np=(k+1)*n). Forward substitution; the diagonal A(i,i) is at AB[i+i*k].
   Exact integer solutions. */

#define TBSV Klas_Tbsv_tbsv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, const float *AB, const float *b,
                  const float *expx)
{
    int np = (k + 1) * n;
    float *gA = dev(AB, np), *gb = dev(b, n), *gx = dev(b, n);
    float x[64];

    TBSV((uint32_t) n, (uint32_t) k, (uint32_t) np, gA, gb, gx);

    MUST(cudaMemcpy(x, gx, n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gb));
    MUST(cudaFree(gx));

    bool t = true;
    for (int i = 0; i < n; i++)
        if (x[i] != expx[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n; i++)
        printf(" %g", x[i]);
    printf("\n");
}

int main()
{
    /* k=0 (diagonal): AB=[2,3,4]; y=[1,2,3] => b=[2,6,12]. */
    float AB0[3] = { 2, 3, 4 };
    float b0[3] = { 2, 6, 12 };
    float x0[3] = { 1, 2, 3 };
    check("tbsv_k0", 3, 0, AB0, b0, x0);

    /* k=1 (bidiagonal), n=4. diag [2,3,4,5], sub [1,1,1];
       AB=[2,1,3,1,4,1,5,G]. y=[1,2,3,4] => b=[2,7,14,23]. */
    float AB1[8] = { 2, 1, 3, 1, 4, 1, 5, 0 };
    float b1[4] = { 2, 7, 14, 23 };
    float x1[4] = { 1, 2, 3, 4 };
    check("tbsv_k1", 4, 1, AB1, b1, x1);

    /* k=2, n=4. AB[i+2j]=A(i,j): A00=1,A10=1,A20=1,A11=2,A21=1,A31=1,A22=3,A32=1,A33=4.
       AB=[1,1,1,2,1,1,3,1,G,4,G,G]. y=ones => b=[1,3,5,6]. */
    float AB2[12] = { 1, 1, 1, 2, 1, 1, 3, 1, 0, 4, 0, 0 };
    float b2[4] = { 1, 3, 5, 6 };
    float x2[4] = { 1, 1, 1, 1 };
    check("tbsv_k2", 4, 2, AB2, b2, x2);

    return !ok;
}
