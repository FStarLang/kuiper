#include <stdio.h>
#include <stdint.h>
#include "GPU_HReduce_U64_Plus.h"

int main()
{
	uint64_t a[1024];
	int i;

	for (i = 0; i < 1024; i++)
		a[i] = i;

	uint64_t r = GPU_HReduce_U64_Plus_reduce(a);

	printf("%lu\n", r);

	return 0;
}
