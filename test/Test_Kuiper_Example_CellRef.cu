#include "Kuiper_Example_CellRef.h"
#include <stdint.h>
#include <stdio.h>

#define N 16

/* Purely CPU-side test of the array1 cell-as-ref API. The extracted
   functions do pointer arithmetic on a plain host buffer:
     cell_get(a, i) == *(a + i)
     cell_set(a, i, w) performs *(a + i) = w
   so we can exercise them on a malloc'd array without any GPU. */

int main()
{
    uint32_t a[N];
    int i;

    /* Reading a cell through a ref must observe the array's contents. */
    for (i = 0; i < N; i++)
        a[i] = (uint32_t) (i * 10);

    for (i = 0; i < N; i++) {
        uint32_t got = Kuiper_Example_CellRef_cell_get(a, (uint32_t) i);
        if (got != (uint32_t) (i * 10)) {
            printf("get error at %d: %u != %u\n", i, got, (unsigned)(i * 10));
            return 1;
        }
    }
    printf("get OK\n");

    /* Writing a cell through the end-to-end setter (full array -> explode
       -> write one cell via a reference setter -> reassemble) must alias
       the array: the underlying buffer is updated and a subsequent get
       reads the new value. */
    for (i = 0; i < N; i++)
        Kuiper_Example_CellRef_array_set_via_ref(a, (uint32_t) i, (uint32_t) (i + 100));

    for (i = 0; i < N; i++) {
        if (a[i] != (uint32_t) (i + 100)) {
            printf("set error at %d: %u != %u\n", i, a[i], (unsigned)(i + 100));
            return 1;
        }
        uint32_t got = Kuiper_Example_CellRef_cell_get(a, (uint32_t) i);
        if (got != (uint32_t) (i + 100)) {
            printf("get-after-set error at %d: %u != %u\n", i, got, (unsigned)(i + 100));
            return 1;
        }
    }
    printf("set OK\n");

    printf("PASS\n");
    return 0;
}
