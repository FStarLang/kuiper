#include <stdio.h>
#include <stdint.h>
#include "Kuiper_HReduce.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

void test(int siz)
{
    float *a;
    float *ga;

    a = (float *)malloc(siz * sizeof a[0]);
    ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);

    int i;

    for (i = 0; i < siz; i++)
        a[i] = i;

    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));

    Kuiper_HReduce_reduce_f32_plus(siz, ga);

    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));

    printf("%f\n", a[0]);
    if (a[0] != siz * (siz - 1) / 2)
        ok = false;
    free(a);
}

int main()
{
    test(510);
    test(511);
    test(512);
    test(513);
    test(514);
    test(1022);
    test(1023);
    test(1024);

    return !ok;
}
