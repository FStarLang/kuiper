#include <stdio.h>
#include <stdint.h>
#include "Klas_Rot.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* c=2, s=3, x[i]=i, y[i]=2i  =>  x'=c*x+s*y=8i,  y'=c*y-s*x=i.
   All exact small integers. */
void test(int siz)
{
    float *a = (float *)malloc((siz ? siz : 1) * sizeof a[0]);
    float *b = (float *)malloc((siz ? siz : 1) * sizeof b[0]);
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], siz);
    int i;
    bool this_ok = true;

    for (i = 0; i < siz; i++) {
        a[i] = (float)i;
        b[i] = 2.0f * (float)i;
    }
    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(float), cudaMemcpyHostToDevice));

    Klas_Rot_rot_f32(2.0f, 3.0f, siz, ga, gb);

    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaMemcpy(b, gb, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));

    for (i = 0; i < siz; i++)
        if (a[i] != 8.0f * (float)i || b[i] != (float)i)
            this_ok = false;
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
    test(2049);
    return !ok;
}
