#ifndef __PULSE__GPU_H
#define __PULSE__GPU_H 1

#include <assert.h>
#include <stdint.h>

#define PULSE_KCALL(foo, nblk, nthr, ...) foo<<<nblk,nthr>>>(__VA_ARGS__)

static inline
uint32_t * PULSE_GPU_ALLOC(size_t len)
{
	uint32_t *ret = NULL;
	if (cudaMalloc(&ret, len) != cudaSuccess)
		assert(0);
	return ret;
}

static inline
void PULSE_GPU_FREE(void *p)
{
	if (cudaFree(p) != cudaSuccess)
		assert(0);
}

static inline
void PULSE_GPU_MEMCPY_H2D(void *p, void *gp, size_t sz)
{
	if (cudaMemcpy(gp, p, sz, cudaMemcpyHostToDevice) != cudaSuccess)
		assert(0);
}

static inline
void PULSE_GPU_MEMCPY_D2H(void *p, void *gp, size_t sz)
{
	if (cudaMemcpy(p, gp, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
		assert(0);
}

#endif
