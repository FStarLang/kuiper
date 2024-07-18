#ifndef __PULSE__GPU_H
#define __PULSE__GPU_H 1

#include <assert.h>
#include <stdint.h>

#define PULSE_KCALL(foo, nblk, nthr, ...)	do {	\
	foo<<<nblk,nthr>>>(__VA_ARGS__);		\
	if (cudaGetLastError() != cudaSuccess)		\
		assert(!"kcall");			\
	} while (0)

static inline
uint32_t * PULSE_GPU_ALLOC(size_t len)
{
	uint32_t *ret = NULL;
	if (cudaMalloc(&ret, len) != cudaSuccess)
		assert(0);
	return ret;
}

#define MUST(e)						({		\
	cudaError_t r = (e);						\
	if (r != cudaSuccess)						\
		assert(!"CALL FAILED: " #e);				\
	})

#endif
