#include <stdio.h>
#include <stdint.h>
#include "Klas_Level1.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* All values are exact small integers, so float comparisons are exact. */
void test(int siz)
{
    float *a = (float *)malloc((siz ? siz : 1) * sizeof a[0]);
    float *b = (float *)malloc((siz ? siz : 1) * sizeof b[0]);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], siz);
    int i;
    bool this_ok = true;

    /* scal: a := 3 * a, with a[i] = i  =>  a[i] = 3i */
    for (i = 0; i < siz; i++)
        a[i] = (float)i;
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_scal_f32(3.0f, siz, ga);
    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (a[i] != 3.0f * (float)i)
            this_ok = false;

    /* axpy: y := 3 * x + y, with x[i] = i, y[i] = 2i  =>  y[i] = 5i */
    for (i = 0; i < siz; i++) {
        a[i] = (float)i;        /* x */
        b[i] = 2.0f * (float)i; /* y */
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_axpy_f32(3.0f, siz, gb, ga);    /* y := 3*x + y */
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (b[i] != 5.0f * (float)i)
            this_ok = false;

    /* copy: y := x, with x[i] = i + 7 */
    for (i = 0; i < siz; i++) {
        a[i] = (float)(i + 7);  /* x */
        b[i] = -1.0f;           /* y */
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_copy_f32(siz, gb, ga);  /* y := x */
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (b[i] != (float)(i + 7))
            this_ok = false;

    /* swap: x <-> y, with x[i] = i, y[i] = -i  =>  x[i] = -i, y[i] = i */
    for (i = 0; i < siz; i++) {
        a[i] = (float)i;        /* x */
        b[i] = -(float)i;       /* y */
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));
    Klas_Level1_swap_f32(siz, ga, gb);
    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < siz; i++)
        if (a[i] != -(float)i || b[i] != (float)i)
            this_ok = false;

    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
    if (!this_ok)
        ok = false;
    printf("test(%d) = %s\n", siz, this_ok ? "ok" : "FAILED");
    free(a);
    free(b);
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
    test(100000);
    return !ok;
}
