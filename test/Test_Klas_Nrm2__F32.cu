#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "Klas_Nrm2.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

/* x[i] = i % 10, so Σ x[i]² stays small (exactly representable). The result
   res ≈ sqrt(Σ x[i]²); we check it within a small relative tolerance, since
   sqrt is irrational. Output is a boolean verdict (deterministic). */
void test(int nth, int siz)
{
    float *a;
    float *ga;

    a = (float *)malloc((siz ? siz : 1) * sizeof a[0]);
    ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], siz);

    int i;
    double sumsq = 0.0;

    for (i = 0; i < siz; i++) {
        a[i] = (float)(i % 10);
        sumsq += (double)(i % 10) * (double)(i % 10);
    }

    MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));

    float res = Klas_Nrm2_nrm2_f32(nth, siz, ga);

    MUST(cudaFree(ga));

    double expected = sqrt(sumsq);
    double err = fabs((double)res - expected);
    bool this_ok = err <= 1e-3 * (expected + 1.0);
    if (!this_ok)
        ok = false;
    printf("test(%d, %d) = %s\n", nth, siz, this_ok ? "ok" : "FAILED");
    free(a);
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
