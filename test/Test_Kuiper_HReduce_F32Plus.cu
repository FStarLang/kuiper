#include <stdio.h>
#include <stdint.h>
#include "Kuiper_HReduce_F32Plus.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	float *a;
	float *ga;
	const size_t siz = Kuiper_HReduce_F32Plus_size;

	a = (float*)malloc(siz * sizeof(float));
	ga = (float*)KPR_GPU_ALLOC(siz * sizeof(float));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));

	Kuiper_HReduce_F32Plus_reduce(siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%lf\n", a[0]);
	free(a);

	return 0;
}
