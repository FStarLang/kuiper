#include "GPU_AtomicReduce.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

#define N 1024

typedef uint64_t u64;

const int sizes[] = {1,2,3,4,5,6,7,8,9, 64, 128, 256, 512, 1024, 0};

int main()
{
	int lap, i, n;
	u64 *a;

	for (lap = 0; sizes[lap]; lap++) {
		n = sizes[lap];
		a = (u64*)malloc(n * sizeof a[0]);

		for (i = 0; i < n; i++)
			a[i] = i;

		// printf("M1\n"); pr(m1); printf("\n");
		// printf("M2\n"); pr(m2); printf("\n");

		u64 r = TIME(GPU_AtomicReduce_reduce(n, a), NULL);
		
		printf("reduce(%d) = %llu\n", n, r);

		free(a);

		return 0;
	}
}
