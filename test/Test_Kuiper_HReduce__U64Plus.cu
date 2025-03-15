#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "Kuiper_HReduce.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	uint64_t *a;
	uint64_t *ga;
	const size_t siz = 1024;

	a  = (uint64_t *)malloc(siz * sizeof a[0]);
	ga = (uint64_t *)KPR_GPU_ALLOC(sizeof ga[0], siz);

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(uint64_t), cudaMemcpyHostToDevice));

	Kuiper_HReduce_reduce_u64_plus(siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%" PRIu64 "\n", a[0]);
	free(a);

	return 0;
}
