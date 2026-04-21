#include "Klas_Softmax.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    half arr[10];
    int i, j;

    for (i = 1; i < 10; i++) {
        printf("len = %d\n", i);
        for (j = 0; j < i; j++) {
            arr[j] = j;
        }
        Klas_Softmax_softmax_f16(i, arr);
        for (j = 0; j < i; j++) {
            printf("%f,", __half2float(arr[j]));
        }
        printf("\n");
    }
    printf("OK\n");

    return 0;
}
