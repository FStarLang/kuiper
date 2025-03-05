#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "Kuiper_HReduceU32Plus.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	uint32_t *a;
	uint32_t *ga;
	const size_t siz = Kuiper_HReduceU32Plus_size;

	a = (uint32_t*)malloc(siz * sizeof(uint32_t));
	ga = (uint32_t*)KPR_GPU_ALLOC(siz * sizeof(uint32_t));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(uint32_t), cudaMemcpyHostToDevice));

	Kuiper_HReduceU32Plus_reduce(siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%" PRIu32 "\n", a[0]);
	free(a);

	return 0;
}
