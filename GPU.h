#ifndef __PULSE__GPU_H
#define __PULSE__GPU_H 1

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

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
		assert(!"cudaMalloc failed");
	return ret;
}

#define MUST(e)						({		\
	cudaError_t uu___r = (e);					\
	if (uu___r != cudaSuccess)					\
		assert(!"CALL FAILED: " #e);				\
	})

#define KRML_HOST_MALLOC            malloc
#define KRML_HOST_CALLOC            calloc
#define KRML_HOST_FREE              free
#define KRML_HOST_IGNORE(x)         (void)(x)
#define KRML_MAYBE_UNUSED_VAR(x)    KRML_HOST_IGNORE(x)
#define KRML_CHECK_SIZE(sz,cnt)     0  // implement!

#endif
