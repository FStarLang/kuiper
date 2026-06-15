#include <stdio.h>
#include <stdint.h>
#include "Klas_Spr.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS spr (lower): AP := AP + alpha*x*x^T, AP the lower triangle of an n x n
   symmetric matrix in PACKED row-major storage (entry (i,j), j<=i, at offset
   i*(i+1)/2 + j; length np = n*(n+1)/2). In place. Each stored entry becomes
   AP[off(i)+j] + alpha*x[i]*x[j]. Exact small integers. */

#define SPR Klas_Spr_spr_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, float alpha, const float *AP0, const float *x,
                  const float *expAP)
{
    int np = n * (n + 1) / 2;
    float *gAP = dev(AP0, np), *gx = dev(x, n);
    float AP[64];

    SPR((uint32_t) n, (uint32_t) np, alpha, gAP, gx);

    MUST(cudaMemcpy(AP, gAP, np * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gAP));
    MUST(cudaFree(gx));

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
    /* AP0 = lower of identity, packed [1, 0,1, 0,0,1]. x=[1,2,3], alpha=1.
       AP[off(i)+j] += x[i]*x[j] -> [2, 2,5, 3,6,10]. */
    float AP3[6] = { 1, 0, 1, 0, 0, 1 };
    float x3[3] = { 1, 2, 3 };
    float e1[6] = { 2, 2, 5, 3, 6, 10 };
    check("spr3_a1", 3, 1.0f, AP3, x3, e1);

    /* same with alpha=2 -> [3, 4,9, 6,12,19]. */
    float AP3b[6] = { 1, 0, 1, 0, 0, 1 };
    float e2[6] = { 3, 4, 9, 6, 12, 19 };
    check("spr3_a2", 3, 2.0f, AP3b, x3, e2);

    /* AP0 packed [5, 2,4], x=[1,2], alpha=1 -> [6, 4,8]. */
    float AP2[3] = { 5, 2, 4 };
    float x2[2] = { 1, 2 };
    float e3[3] = { 6, 4, 8 };
    check("spr2_a1", 2, 1.0f, AP2, x2, e3);

    return !ok;
}
