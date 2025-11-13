#ifndef __KUIPER_H
#define __KUIPER_H 1

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "atomics.h"
#include "vectorops.h"
#if (!defined(KUIPER_CFG_TENSORCORES) || KUIPER_CFG_TENSORCORES)
#include "tensorcores.h"
#endif

typedef half half_t; /* crutch */
/* NOTE: making this a macro means it works in host/device, but we
 * need a more scalable solution. */
#define __hexp(f) (__float2half(exp(__half2float(f))))

static inline
void __MUST(cudaError_t rc, const char * str, const char * func, const char *fname, int line)
{
	if (rc != cudaSuccess) {
		fprintf(stderr, "*** ABORTING ***\n");
		fprintf(stderr, " This call failed: %s\n", str);
		fprintf(stderr, " At file %s, line %d\n", fname, line);
		fprintf(stderr, " In function %s\n", func);
		fprintf(stderr, " CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}
}

/*
 * All kernel calls extract to this. The shared memory will just
 * be zero if not used, etc.
 */
#define KPR_KCALL(foo, nblk, nthr, e_size, ...)						\
	do {										\
		auto _nblk = (nblk);							\
		auto _nthr = (nthr);							\
		if (_nblk > 0 && _nthr > 0) {						\
			cudaStream_t fresh;						\
			cudaStreamCreate(&fresh);					\
			foo<<<_nblk, _nthr, (e_size), fresh>>>(__VA_ARGS__);		\
			__MUST(cudaGetLastError(), "kcall", __func__, __FILE__, __LINE__);\
		}									\
	} while (0)

#define KPR_SHMEM()									\
	({										\
		extern __shared__ char a[];						\
		(char*)a;								\
	})

#define KPR_SHMEM_AT(off)								\
	(void*)(KPR_SHMEM() + (off))

#define FStar_Pervasives_Native_fst(x) x

#define KPR_GUARD(b)								\
	do {									\
		if (!(b)) {							\
			fprintf(stderr, "*** ABORTING ***\n");			\
			fprintf(stderr, " Guard failed: %s\n", #b);		\
			fprintf(stderr, " In function %s\n", __func__);		\
			fprintf(stderr, " At " __FILE__ ":%d\n", __LINE__);	\
			abort();						\
		}								\
	} while(0)

#ifndef NDEBUG
#define KPR_ASSERT(b)								\
	do {									\
		if (!(b)) {							\
			fprintf(stderr, "*** ABORTING ***\n");			\
			fprintf(stderr, " Assertion failed: %s\n", #b);		\
			fprintf(stderr, " In function %s\n", __func__);		\
			fprintf(stderr, " At " __FILE__ ":%d\n", __LINE__);	\
			abort();						\
		}								\
	} while(0)
#else
#define KPR_ASSERT(b)
#endif

#define MUST(e)			__MUST(e, #e, __func__, __FILE__, __LINE__)

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
#define KRML_HOST_EPRINTF(s, ...)   fprintf(stderr, __VA_ARGS__)
#define KRML_HOST_EXIT(rc)          exit(rc)

static inline
void * __KPR_GPU_ALLOC(size_t sz, size_t len, const char * func, const char *str, const char *fname,
			     int line)
{
	void *ret = NULL;
	KRML_CHECK_SIZE(sz, len);
	__MUST(cudaMalloc(&ret, sz*len), str, func, fname, line);
	return ret;
}

#define KPR_GPU_ALLOC(sz, len)						\
	__KPR_GPU_ALLOC(sz, len, "KPR_GPU_ALLOC(" #sz ", " #len ")", __func__, __FILE__, __LINE__)

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

#define KPR_SHMEM_FITS(e) KPR_ASSERT((e) <= 101376) // 99KiB

#endif /* __KUIPER_H */
