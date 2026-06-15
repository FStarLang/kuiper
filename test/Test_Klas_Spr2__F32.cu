#include <stdio.h>
#include <stdint.h>
#include "Klas_Spr2.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS spr2 (lower): AP := AP + alpha*(x*y^T + y*x^T), AP the lower triangle
   of an n x n symmetric matrix in PACKED row-major storage (entry (i,j), j<=i,
   at offset i*(i+1)/2 + j; length np = n*(n+1)/2). In place. Each stored entry
   becomes AP[off(i)+j] + alpha*(x[i]*y[j] + y[i]*x[j]). Exact small integers. */

#define SPR2 Klas_Spr2_spr2_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, float alpha, const float *AP0, const float *x,
                  const float *y, const float *expAP)
{
    int np = n * (n + 1) / 2;
    float *gAP = dev(AP0, np), *gx = dev(x, n), *gy = dev(y, n);
    float AP[64];

    SPR2((uint32_t) n, (uint32_t) np, alpha, gAP, gx, gy);

    MUST(cudaMemcpy(AP, gAP, np * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gAP));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));

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
    /* AP0 = lower of identity packed [1, 0,1, 0,0,1]. x=[1,0,1], y=[0,1,1], a=1.
       AP[off(i)+j] += x[i]*y[j]+y[i]*x[j] -> [1, 1,1, 1,1,3]. */
    float AP3[6] = { 1, 0, 1, 0, 0, 1 };
    float x3[3] = { 1, 0, 1 };
    float y3[3] = { 0, 1, 1 };
    float e1[6] = { 1, 1, 1, 1, 1, 3 };
    check("spr2_3", 3, 1.0f, AP3, x3, y3, e1);

    /* AP0 = lower identity packed [1, 0,1]. x=[1,2], y=[3,4], alpha=1.
       -> [1+6, 0+10, 1+16] = [7, 10, 17]. */
    float AP2[3] = { 1, 0, 1 };
    float x2[2] = { 1, 2 };
    float y2[2] = { 3, 4 };
    float e2[3] = { 7, 10, 17 };
    check("spr2_2", 2, 1.0f, AP2, x2, y2, e2);

    return !ok;
}
