#include "Kuiper_ArrayReversal.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#define N 1024

typedef uint64_t u64;

// Only even, >=2
const int sizes[] = { 2, 4, 6, 8, 64, 128, 256, 512, 1024, 0 };

int main()
{
	int lap, i, n;
	u64 *a;

	for (lap = 0; sizes[lap]; lap++) {
		n = sizes[lap];
		printf("Trying n = %i\n", n);

		a = (u64 *) malloc(n * sizeof a[0]);
		u64 *aa = (u64 *) KPR_GPU_ALLOC(n * sizeof aa[0]);

		for (i = 0; i < n; i++)
			a[i] = i;

		MUST(cudaMemcpy(aa, a, n * 8U, cudaMemcpyHostToDevice));

		TIME_void(Kuiper_ArrayReversal_reverse_u64(n, aa), NULL);

		MUST(cudaMemcpy(a, aa, n * 8U, cudaMemcpyDeviceToHost));

		for (i = 0; i < n; i++) {
			if (a[i] != n - i - 1) {
				printf("Error at %d: %" PRIu64 " != %" PRIu64
				       "\n", i, a[i], n - i - 1);
				return 1;
			}
		}
		printf("OK %i\n", n);

		free(a);
	}
	return 0;
}
