#include <stdio.h>
#include <stdint.h>
#include "Klas_Syrk.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS syrk (lower, trans=N): X := alpha*A*A^T + beta*C. A is n x k row-major
   (length n*k); C, X are n x n row-major (A*A^T is symmetric). Out-of-place
   output X. Each cell is a dot of two rows of A. Exact small integers. */

#define SYRK Klas_Syrk_syrk_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, int k, float alpha, float beta, const float *A,
                  const float *C, const float *expX)
{
    float *gA = dev(A, n * k), *gC = dev(C, n * n), *gX = dev(C, n * n);
    float X[64];

    SYRK((uint32_t) n, (uint32_t) k, alpha, beta, gA, gC, gX);

    MUST(cudaMemcpy(X, gX, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gC));
    MUST(cudaFree(gX));

    bool t = true;
    for (int i = 0; i < n * n; i++)
        if (X[i] != expX[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n * n; i++)
        printf(" %g", X[i]);
    printf("\n");
}

int main()
{
    /* A = [[1,2,3],[4,5,6]] (n=2, k=3, row-major).
       A*A^T = [[14,32],[32,77]]. C = ones. */
    float A1[6] = { 1, 2, 3, 4, 5, 6 };
    float C1[4] = { 1, 1, 1, 1 };
    float X1[4] = { 15, 33, 33, 78 };   /* alpha=1, beta=1 */
    check("syrk2_11", 2, 3, 1.0f, 1.0f, A1, C1, X1);

    float X1b[4] = { 28, 64, 64, 154 }; /* alpha=2, beta=0 */
    check("syrk2_20", 2, 3, 2.0f, 0.0f, A1, C1, X1b);

    /* A = [[1,0],[0,1],[1,1]] (n=3, k=2). A*A^T = [[1,0,1],[0,1,1],[1,1,2]]. */
    float A2[6] = { 1, 0, 0, 1, 1, 1 };
    float C2[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float X2[9] = { 1, 0, 1, 0, 1, 1, 1, 1, 2 };        /* alpha=1, beta=0 */
    check("syrk3_10", 3, 2, 1.0f, 0.0f, A2, C2, X2);

    return !ok;
}
