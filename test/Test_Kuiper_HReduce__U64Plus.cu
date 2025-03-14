#include <stdio.h>
#include <stdint.h>
#include "Kuiper_HReduce_U64Plus.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	uint64_t *a;
	uint64_t *ga;
	const size_t siz = Kuiper_HReduce_U64Plus_size;

	a = (uint64_t*)malloc(siz * sizeof(uint64_t));
	ga = (uint64_t*)KPR_GPU_ALLOC(siz * sizeof(uint64_t));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(uint64_t), cudaMemcpyHostToDevice));

	Kuiper_HReduce_U64Plus_reduce(siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%lu\n", a[0]);
	free(a);

	return 0;
}
