#ifndef __PULSE__GPU_H
#define __PULSE__GPU_H 1

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#define PULSE_KCALL(foo, nblk, nthr, ...)				\
	do {								\
		foo<<<nblk,nthr>>>(__VA_ARGS__);			\
		if (cudaGetLastError() != cudaSuccess)			\
			assert(!"kcall");				\
	} while (0)

#define MUST(e)								\
	({								\
		cudaError_t uu___r = (e);				\
		if (uu___r != cudaSuccess) {				\
			fprintf(stderr, "CALL FAILED: " #e "\n");	\
			fprintf(stderr, "CUDA error: %s\n",		\
					cudaGetErrorString(uu___r));	\
			exit(1);					\
		}							\
	})

static inline
uint32_t * PULSE_GPU_ALLOC(size_t len)
{
	uint32_t *ret = NULL;
	MUST(cudaMalloc(&ret, len));
	return ret;
}

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
