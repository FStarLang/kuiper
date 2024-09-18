#ifndef __KUIPER_H
#define __KUIPER_H 1

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "krml_float.h"
#include "atomics.h"

#define blockIdx_x() blockIdx.x
#define blockDim_x() blockDim.x
#define threadIdx_x() threadIdx.x

#include <cuda_runtime.h>

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

#define KPR_KCALL_SHMEM_ASYNC(foo, nblk, nthr, e_size, cnt, ...)		\
	do {									\
		foo<<<nblk, nthr, ((e_size) * (cnt))>>>(__VA_ARGS__);		\
		__MUST(cudaGetLastError(), "kcall", __FILE__, __LINE__);	\
	} while (0)

#define KPR_KCALL_SHMEM(foo, nblk, nthr, e_size, cnt, ...)			\
	do {									\
		KPR_KCALL_SHMEM_ASYNC(foo, nblk, nthr, e_size, cnt, __VA_ARGS__);\
		cudaDeviceSynchronize();					\
	} while(0)

#define KPR_KCALL_ASYNC(foo, nblk, nthr, ...)					\
	KPR_KCALL_SHMEM_ASYNC(foo, nblk, nthr, 0, 0, __VA_ARGS__)

#define KPR_KCALL(foo, nblk, nthr, ...)					\
	KPR_KCALL_SHMEM(foo, nblk, nthr, 0, 0, __VA_ARGS__)

#define KPR_SHMEM()							\
	({								\
		extern __shared__ char a[];				\
		a;							\
	})

#define MUST(e)								\
	__MUST(e, #e, __FILE__, __LINE__)

static inline
void * __KPR_GPU_ALLOC(size_t len, const char *str, const char *fname,
			     int line)
{
	void *ret = NULL;
	__MUST(cudaMalloc(&ret, len), str, fname, line);
	return ret;
}

#define KPR_GPU_ALLOC(len)						\
	__KPR_GPU_ALLOC(len, "KPR_GPU_ALLOC(" #len ")", __FILE__, __LINE__)

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

static inline
void INFO ()
{
	printf("sizeof(short) = %lu\n", sizeof(short));
	printf("sizeof(int) = %lu\n", sizeof(int));
	printf("sizeof(long) = %lu\n", sizeof(long));
	printf("sizeof(long long) = %lu\n", sizeof(long long));

	printf("sizeof(unsigned short) = %lu\n", sizeof(unsigned short));
	printf("sizeof(unsigned int) = %lu\n", sizeof(unsigned int));
	printf("sizeof(unsigned long) = %lu\n", sizeof(unsigned long));
	printf("sizeof(unsigned long long) = %lu\n", sizeof(unsigned long long));
}

#define FStar_Pervasives_Native___proj__Mktuple3__item___1(i) ((i).fst)
#define FStar_Pervasives_Native___proj__Mktuple3__item___2(i) ((i).snd)
#define FStar_Pervasives_Native___proj__Mktuple3__item___3(i) ((i).thd)

#endif
