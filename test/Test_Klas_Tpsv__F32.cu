#include <stdio.h>
#include <stdint.h>
#include "Klas_Tpsv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS tpsv (lower, non-unit): solve A*y = b for y, A n x n lower-triangular
   in PACKED row-major storage (entry (i,j), j<=i, at offset i*(i+1)/2 + j; AP
   has length np = n*(n+1)/2). Forward substitution. Exact integer solutions. */

#define TPSV Klas_Tpsv_tpsv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, const float *AP, const float *b, const float *expx)
{
    int np = n * (n + 1) / 2;
    float *gA = dev(AP, np), *gb = dev(b, n), *gx = dev(b, n);
    float x[64];

    TPSV((uint32_t) n, (uint32_t) np, gA, gb, gx);

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
    /* A=[[2,.,.],[1,3,.],[4,5,1]] packed [2, 1,3, 4,5,1]. y=[2,3,4] => b=[4,11,27]. */
    float AP3[6] = { 2, 1, 3, 4, 5, 1 };
    float b3[3] = { 4, 11, 27 };
    float x3[3] = { 2, 3, 4 };
    check("tpsv3", 3, AP3, b3, x3);

    /* A=[[5,.],[2,4]] packed [5, 2,4]. y=[1,2] => b=[5,10]. */
    float AP2[3] = { 5, 2, 4 };
    float b2[2] = { 5, 10 };
    float x2[2] = { 1, 2 };
    check("tpsv2", 2, AP2, b2, x2);

    /* A=[[1,.,.,.],[0,2,.,.],[0,0,1,.],[1,1,1,1]] packed
       [1, 0,2, 0,0,1, 1,1,1,1]. y=[1,2,3,4] => b=[1,4,3,10]. */
    float AP4[10] = { 1, 0, 2, 0, 0, 1, 1, 1, 1, 1 };
    float b4[4] = { 1, 4, 3, 10 };
    float x4[4] = { 1, 2, 3, 4 };
    check("tpsv4", 4, AP4, b4, x4);

    return !ok;
}
