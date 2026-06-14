#include <stdio.h>
#include <stdint.h>
#include "Klas_Geam.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* alpha=3, beta=5, A[i]=i, B[i]=2i  =>  C[i] = 3i + 10i = 13i.
   All exact small integers. */
void test(int len)
{
    float *a = (float *)malloc((len ? len : 1) * sizeof a[0]);
    float *b = (float *)malloc((len ? len : 1) * sizeof b[0]);
    float *c = (float *)malloc((len ? len : 1) * sizeof c[0]);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], len);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], len);
    float *gc = (float *)KPR_GPU_ALLOC(sizeof gc[0], len);
    int i;
    bool this_ok = true;

    for (i = 0; i < len; i++) {
        a[i] = (float)i;
        b[i] = 2.0f * (float)i;
        c[i] = -1.0f;
    }
    MUST(cudaMemcpy(ga, a, len * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, len * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gc, c, len * sizeof(float), cudaMemcpyHostToDevice));

    Klas_Geam_geam_f32(3.0f, 5.0f, len, gc, ga, gb);

    MUST(cudaMemcpy(c, gc, len * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    MUST(cudaFree(gc));

    for (i = 0; i < len; i++)
        if (c[i] != 13.0f * (float)i)
            this_ok = false;
    if (!this_ok)
        ok = false;
    printf("test(%d) = %s\n", len, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
    free(c);
}

int main()
{
    test(0);
    test(1);
    test(2);
    test(511);
    test(512);
    test(513);
    test(1024);
    test(1025);
    test(2048);
    return !ok;
}
