#include <stdio.h>
#include <stdint.h>
#include "GPU_HReduceU64Plus.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	uint64_t *a;
	uint64_t *ga;
	const size_t siz = GPU_HReduceU64Plus_size;

	a = (uint64_t*)malloc(siz * sizeof(uint64_t));
	ga = (uint64_t*)PULSE_GPU_ALLOC(siz * sizeof(uint64_t));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(uint64_t), cudaMemcpyHostToDevice));

	PULSE_KCALL(GPU_HReduceU64Plus_k_reduce, 1, siz, siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%lu\n", a[0]);
	free(a);

	return 0;
}
