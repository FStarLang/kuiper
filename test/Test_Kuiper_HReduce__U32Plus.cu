#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "Kuiper_HReduce.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

void test(int siz)
{
    uint32_t *a;
    uint32_t *ga;

    a = (uint32_t *) malloc(siz * sizeof a[0]);
    ga = (uint32_t *) KPR_GPU_ALLOC(sizeof ga[0], siz);

    int i;

    for (i = 0; i < siz; i++)
        a[i] = i;

    MUST(cudaMemcpy(ga, a, siz * sizeof(uint32_t), cudaMemcpyHostToDevice));

    Kuiper_HReduce_reduce_u32_plus(siz, ga);

    MUST(cudaMemcpy(a, ga, siz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));

    printf("%" PRIu32 "\n", a[0]);
    if (a[0] != siz*(siz-1)/2)
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
