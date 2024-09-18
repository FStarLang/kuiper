#include <stdio.h>
#include <stdint.h>
#include "Kuiper_HReduceF32Plus.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	float *a;
	float *ga;
	const size_t siz = Kuiper_HReduceF32Plus_size;

	a = (float*)malloc(siz * sizeof(float));
	ga = (float*)KPR_GPU_ALLOC(siz * sizeof(float));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(float), cudaMemcpyHostToDevice));

	KPR_KCALL(Kuiper_HReduceF32Plus_k_reduce, 1, siz, siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(float), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%lf\n", a[0]);
	free(a);

	return 0;
}
