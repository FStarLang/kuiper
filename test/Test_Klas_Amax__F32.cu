#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include "Klas_Amax.h"

bool ok = true;

/* It would be nicer to write a purely-Pulse test. */

#define TYPE float
#define FUN  Klas_Amax_amax_f32

/* Reference: 0-based index of the first element of largest absolute value
   (matches Klas.Amax.is_amax: strict '>' switch keeps the earliest maximizer). */
uint64_t cpu_amax(TYPE *a, int siz)
{
    int best = 0;
    for (int i = 1; i < siz; i++)
        if (fabsf(a[i]) > fabsf(a[best]))
            best = i;
    return (uint64_t) best;
}

/* pat selects a fill pattern; all values are exact small integers. */
void test(int siz, int pat)
{
    TYPE *a = (TYPE *) malloc((siz ? siz : 1) * sizeof a[0]);
    TYPE *ga = (TYPE *) KPR_GPU_ALLOC(sizeof ga[0], siz);

    for (int i = 0; i < siz; i++) {
        switch (pat) {
        case 0:
            a[i] = (TYPE) i;
            break;              /* |a| increasing  -> last  */
        case 1:
            a[i] = -(TYPE) i;
            break;              /* |a| increasing  -> last  */
        case 2:
            a[i] = (TYPE) (siz - i);
            break;              /* |a| decreasing  -> first */
        case 3:
            a[i] = 7.0f;
            break;              /* all equal       -> first */
        default:
            a[i] = (i == siz / 2) ? 1000.0f : (TYPE) (i % 10);  /* mid peak               */
        }
    }

    MUST(cudaMemcpy(ga, a, siz * sizeof(TYPE), cudaMemcpyHostToDevice));
    uint64_t res = FUN((uint32_t) siz, ga);
    MUST(cudaFree(ga));

    uint64_t expected = cpu_amax(a, siz);
    if (res != expected)
        ok = false;
    printf("test(%d, %d) = %" PRIu64 "%s\n", siz, pat, res, ok ? "" : " (FAILED)");
    free(a);
}

int main()
{
    for (int pat = 0; pat < 5; pat++) {
        test(1, pat);
        test(2, pat);
        test(3, pat);
        test(17, pat);
        test(512, pat);
        test(1000, pat);
    }
    return !ok;
}
