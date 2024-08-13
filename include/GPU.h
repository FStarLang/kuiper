#ifndef __PULSE__GPU_H
#define __PULSE__GPU_H 1

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "krml_float.h"

#include <cuda_runtime.h>

#define PULSE_KCALL(foo, nblk, nthr, ...)				\
	do {								\
		foo<<<nblk,nthr>>>(__VA_ARGS__);			\
		if (cudaGetLastError() != cudaSuccess)			\
			assert(!"kcall");				\
	} while (0)

static inline
void __MUST(cudaError_t rc, const char * str, const char *fname, int line)
{
	if (rc != cudaSuccess) {
		fprintf(stderr, "*** ABORTING ***\n");
		fprintf(stderr, "This call failed: %s\n", str);
		fprintf(stderr, "At file %s, line %d\n", fname, line);
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}
}

#define MUST(e)								\
	__MUST(e, #e, __FILE__, __LINE__)

static inline
uint32_t * __PULSE_GPU_ALLOC(size_t len, const char *str, const char *fname,
			     int line)
{
	uint32_t *ret = NULL;
	__MUST(cudaMalloc(&ret, len), str, fname, line);
	return ret;
}

#define PULSE_GPU_ALLOC(len)						\
	__PULSE_GPU_ALLOC(len, "PULSE_GPU_ALLOC(" #len ")", __FILE__, __LINE__)

#define KRML_HOST_MALLOC            malloc
#define KRML_HOST_CALLOC            calloc
#define KRML_HOST_FREE              free
#define KRML_HOST_IGNORE(x)         (void)(x)
#define KRML_MAYBE_UNUSED_VAR(x)    KRML_HOST_IGNORE(x)
#define KRML_CHECK_SIZE(size_elt, sz)					\
	do {								\
		if (((size_t)(sz)) > ((size_t)(SIZE_MAX / (size_elt))))	\
			assert(!"CHECK_SIZE");				\
	} while (0)

#endif
