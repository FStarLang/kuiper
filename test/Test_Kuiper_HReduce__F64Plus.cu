#include <stdio.h>
#include <stdint.h>
#include "Kuiper_HReduce.h"

/* It would be nicer to write a purely-Pulse test. */
int main()
{
	double *a;
	double *ga;
	const size_t siz = 1024;

	a = (double *)malloc(siz * sizeof(double));
	ga = (double *)KPR_GPU_ALLOC(siz * sizeof(double));

	int i;

	for (i = 0; i < siz; i++)
		a[i] = i;

	MUST(cudaMemcpy(ga, a, siz * sizeof(double), cudaMemcpyHostToDevice));

	Kuiper_HReduce_reduce_f64_plus(siz, ga);

	MUST(cudaMemcpy(a, ga, siz * sizeof(double), cudaMemcpyDeviceToHost));
	MUST(cudaFree(ga));

	printf("%lf\n", a[0]);
	free(a);

	return 0;
}
