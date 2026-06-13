#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "Klas_Dot.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

#define TYPE float
#define FUN  Klas_Dot_dot_f32
#define PR   "%f"

/* x[i] = i, y[i] = 2  =>  dot = Σ 2i = siz*(siz-1).
   All values are exact small integers (results stay below 2^24). */
void test(int nth, int siz)
{
    TYPE *a;
    TYPE *b;
    TYPE *ga;
    TYPE *gb;

    a = (TYPE *) malloc((siz ? siz : 1) * sizeof a[0]);
    b = (TYPE *) malloc((siz ? siz : 1) * sizeof b[0]);
    ga = (TYPE *) KPR_GPU_ALLOC(sizeof ga[0], siz);
    gb = (TYPE *) KPR_GPU_ALLOC(sizeof gb[0], siz);

    int i;

    for (i = 0; i < siz; i++) {
        a[i] = (TYPE) i;
        b[i] = 2.0f;
    }

    MUST(cudaMemcpy(ga, a, siz * sizeof(TYPE), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gb, b, siz * sizeof(TYPE), cudaMemcpyHostToDevice));

    TYPE res = FUN(nth, siz, ga, gb);

    MUST(cudaFree(ga));
    MUST(cudaFree(gb));

    // Note: assuming no FP error
    if (res != (TYPE) (siz * (siz - 1)))
        ok = false;
    printf("test(%d, %d) = " PR "%s\n", nth, siz, res, ok ? "" : " (FAILED)");
    free(a);
    free(b);
}

int main()
{
    /* Tests with full blocks. */
    test(1024, 510);
    test(1024, 511);
    test(1024, 512);
    test(1024, 513);
    test(1024, 1023);
    test(1024, 1024);
    test(1024, 1025);
    test(1024, 2048);
    test(1024, 2049);

    /* Smaller blocks */
    test(512, 512);
    test(512, 513);
    test(512, 1024);

    /* Weird sizes */
    test(1, 0);
    test(1, 1);
    test(1, 2);
    test(2, 0);
    test(2, 1);
    test(3, 2);
    test(4, 2);

    return !ok;
}
