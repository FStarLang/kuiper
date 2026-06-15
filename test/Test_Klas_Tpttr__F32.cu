#include <stdio.h>
#include <stdint.h>
#include "Klas_Tpttr.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS tpttr (lower): unpack a PACKED lower-triangular matrix (AP, length
   np = n*(n+1)/2, entry (i,j) j<=i at offset i*(i+1)/2 + j) into a full n x n
   row-major matrix with the strict upper triangle set to zero:
   A_full[i*n+j] = (j<=i) ? AP[off(i)+j] : 0. Exact small integers. */

#define TPTTR Klas_Tpttr_tpttr_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, const float *AP, const float *expA)
{
    int np = n * (n + 1) / 2;
    float *gA = dev(AP, np), *gX = dev(AP, n * n > np ? n * n : np);
    float A[64];

    TPTTR((uint32_t) n, (uint32_t) np, gA, gX);

    MUST(cudaMemcpy(A, gX, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gA));
    MUST(cudaFree(gX));

    bool t = true;
    for (int i = 0; i < n * n; i++)
        if (A[i] != expA[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n * n; i++)
        printf(" %g", A[i]);
    printf("\n");
}

int main()
{
    /* packed [2, 1,3, 4,5,6] -> full [[2,0,0],[1,3,0],[4,5,6]]. */
    float AP3[6] = { 2, 1, 3, 4, 5, 6 };
    float A3[9] = { 2, 0, 0, 1, 3, 0, 4, 5, 6 };
    check("tpttr3", 3, AP3, A3);

    /* packed [5, 2,4] -> full [[5,0],[2,4]]. */
    float AP2[3] = { 5, 2, 4 };
    float A2[4] = { 5, 0, 2, 4 };
    check("tpttr2", 2, AP2, A2);

    /* packed [1, 2,3, 4,5,6, 7,8,9,10] -> 4x4 lower. */
    float AP4[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    float A4[16] = { 1, 0, 0, 0, 2, 3, 0, 0, 4, 5, 6, 0, 7, 8, 9, 10 };
    check("tpttr4", 4, AP4, A4);

    return !ok;
}
