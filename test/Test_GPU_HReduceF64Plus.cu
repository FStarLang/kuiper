#include <stdio.h>
#include <stdint.h>
#include "GPU_HReduceF64Plus.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	double *a;
	double *ga;
	const size_t siz = GPU_HReduceF64Plus_size;

	a = (double*)malloc(siz * sizeof(double));
	ga = (double*)PULSE_GPU_ALLOC(siz * sizeof(double));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(double), cudaMemcpyHostToDevice));

	PULSE_KCALL(GPU_HReduceF64Plus_k_reduce, 1, siz, siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(double), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%lf\n", a[0]);
	free(a);

	return 0;
}
