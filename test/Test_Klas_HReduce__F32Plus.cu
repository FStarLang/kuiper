#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "Klas_HReduce.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

#define TYPE float
#define FUN  Klas_HReduce_reduce_f32_plus
#define PR   "%f"

void test(int nth, int siz)
{
    TYPE *a;
    TYPE *ga;

    a = (TYPE *)malloc(siz * sizeof a[0]);
    ga = (TYPE *)KPR_GPU_ALLOC(sizeof ga[0], siz);

    int i;

    for (i = 0; i < siz; i++)
        a[i] = i;

    MUST(cudaMemcpy(ga, a, siz * sizeof(TYPE), cudaMemcpyHostToDevice));

    TYPE res = FUN(nth, siz, ga);

    MUST(cudaMemcpy(a, ga, siz * sizeof(TYPE), cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));

    // Note: assuming no FP error
    if (res != siz * (siz - 1) / 2)
        ok = false;
    printf("test(%d, %d) = " PR "%s\n", nth, siz, res, ok ? "" : " (FAILED)");
    free(a);
}

int main()
{
    /* Tests with full blocks. */
    test(1024, 510);
    test(1024, 511);
    test(1024, 512);
    test(1024, 513);
    test(1024, 514);
    test(1024, 1022);
    test(1024, 1023);
    test(1024, 1024);
    test(1024, 1025);
    test(1024, 2048);
    test(1024, 2049);

    /* Smaller blocks */
    test(512, 510);
    test(512, 511);
    test(512, 512);
    test(512, 513);
    test(512, 514);
    test(512, 1022);
    test(512, 1023);
    test(512, 1024);
    test(512, 1025);
    test(512, 2048);

    /* Weird sizes */
    test(1, 0);
    test(1, 1);
    test(1, 2);
    test(2, 0);
    test(2, 1);
    test(2, 2);
    test(3, 0);
    test(3, 1);
    test(3, 2);
    test(4, 0);
    test(4, 1);
    test(4, 2);

    return !ok;
}
