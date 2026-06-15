#include <stdio.h>
#include <stdint.h>
#include "Klas_Trttp.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS trttp (lower): pack the lower triangle of a full n x n row-major matrix
   into PACKED row-major form: AP[off(i)+j] = A_full[i*n+j] for j<=i, where
   off(i)=i*(i+1)/2 and AP has length np = n*(n+1)/2. The strict upper triangle
   of the input is filled with garbage to confirm it is not read. */

#define TRTTP Klas_Trttp_trttp_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, const float *A, const float *expAP)
{
    int np = n * (n + 1) / 2;
    float *gX = dev(A, n * n), *gAP = dev(A, n * n > np ? n * n : np);
    float AP[64];

    TRTTP((uint32_t) n, (uint32_t) np, gX, gAP);

    MUST(cudaMemcpy(AP, gAP, np * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gX));
    MUST(cudaFree(gAP));

    bool t = true;
    for (int i = 0; i < np; i++)
        if (AP[i] != expAP[i])
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < np; i++)
        printf(" %g", AP[i]);
    printf("\n");
}

int main()
{
    /* full [[2,.,.],[1,3,.],[4,5,6]] (upper garbage) -> packed [2, 1,3, 4,5,6]. */
    float A3[9] = { 2, 99, 99, 1, 3, 99, 4, 5, 6 };
    float AP3[6] = { 2, 1, 3, 4, 5, 6 };
    check("trttp3", 3, A3, AP3);

    /* full [[5,.],[2,4]] -> packed [5, 2,4]. */
    float A2[4] = { 5, 99, 2, 4 };
    float AP2[3] = { 5, 2, 4 };
    check("trttp2", 2, A2, AP2);

    /* 4x4 lower -> packed [1, 2,3, 4,5,6, 7,8,9,10]. */
    float A4[16] = { 1, 99, 99, 99, 2, 3, 99, 99, 4, 5, 6, 99, 7, 8, 9, 10 };
    float AP4[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    check("trttp4", 4, A4, AP4);

    return !ok;
}
