#include <stdio.h>
#include <stdint.h>
#include "Kuiper_HReduce.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
    float *a;
    float *ga;
    const size_t siz = 1024;

    a = (float *)malloc(siz * sizeof a[0]);
    ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);

    int i;

    for (i = 0; i < siz; i++)
        a[i] = i;

    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));

    Kuiper_HReduce_reduce_f32_plus(siz, ga);

    MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));

    printf("%lf\n", a[0]);
    free(a);

    return 0;
}
