#include <stdio.h>
#include <stdint.h>
#include "Klas_Trsv.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */
/* cuBLAS trsv: solve A*x = b, A lower-triangular (non-unit diagonal), row-major
   n x n; b, x length n. Chosen so the solution is exact small integers. */

#define TRSV Klas_Trsv_trsv_f32

static float *dev(const float *h, int n)
{
    float *g = (float *)KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(g, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return g;
}

static void check(const char *name, int n, const float *A, const float *b, const float *expx)
{
    float *gA = dev(A, n * n), *gb = dev(b, n), *gx = dev(b, n);
    float x[64];

    TRSV((uint32_t) n, gA, gb, gx);

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
    /* 2x2: [[5,0],[2,4]] x=[1,2] => b=[5,10] */
    float A2[4] = { 5, 0, 2, 4 };
    float b2[2] = { 5, 10 };
    float x2[2] = { 1, 2 };
    check("trsv2", 2, A2, b2, x2);

    /* diagonal 2x2: [[3,0],[0,7]] x=[2,2] => b=[6,14] */
    float Ad[4] = { 3, 0, 0, 7 };
    float bd[2] = { 6, 14 };
    float xd[2] = { 2, 2 };
    check("trsv_diag", 2, Ad, bd, xd);

    /* 3x3: [[2,0,0],[1,3,0],[4,5,1]] x=[2,3,4] => b=[4,11,27] */
    float A3[9] = { 2, 0, 0, 1, 3, 0, 4, 5, 1 };
    float b3[3] = { 4, 11, 27 };
    float x3[3] = { 2, 3, 4 };
    check("trsv3", 3, A3, b3, x3);

    /* 4x4 with unit-ish diagonal: x=[1,2,3,4] */
    float A4[16] = {
        1, 0, 0, 0,
        2, 1, 0, 0,
        0, 3, 1, 0,
        1, 1, 1, 1
    };
    float x4[4] = { 1, 2, 3, 4 };
    /* b = A4 * x4 */
    float b4[4] = {
        1 * 1,                  /* 1 */
        2 * 1 + 1 * 2,          /* 4 */
        3 * 2 + 1 * 3,          /* 9 */
        1 * 1 + 1 * 2 + 1 * 3 + 1 * 4   /* 10 */
    };
    check("trsv4", 4, A4, b4, x4);

    return !ok;
}
