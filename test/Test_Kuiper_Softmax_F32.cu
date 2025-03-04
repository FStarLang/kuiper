#include "Kuiper_Softmax_F32.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	float arr[10];
        int i, j;

        for (i = 1; i < 10; i++) {
                printf("len = %d\n", i);
                for (j = 0; j < i; j++) {
                        arr[j] = j;
                }
                Kuiper_Softmax_F32_softmax(i, arr);
                for (j = 0; j < i; j++) {
                        printf("%f ", arr[j]);
                }
                printf("\n");
        }
	printf("OK\n");

	return 0;
}
