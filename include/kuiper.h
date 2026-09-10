#ifndef KUIPER_H
#define KUIPER_H 1

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* Make Custard's custard_f16/custard_bf16 be CUDA's own 16-bit types, which
   is what wmma::fragment is a template over.  Must precede the generated
   header's narrow-float support block. */
#include "kuiper/custard_f16_cuda.h"
#include "kuiper/atomics.h"
#include "kuiper/vectorops.h"
#include "kuiper/math.h"

#if (!defined(KUIPER_CFG_TENSORCORES) || KUIPER_CFG_TENSORCORES)
#include "kuiper/tensorcores.h"
#endif

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
#define KPR_KCALL(foo, nblk, nthr, e_size, stream, ...)						\
	do {										\
		auto _nblk = (nblk);							\
		auto _nthr = (nthr);							\
		if (_nblk > 0 && _nthr > 0) {						\
			foo<<<_nblk, _nthr, (e_size), stream>>>(__VA_ARGS__);\
			__MUST(cudaGetLastError(), "kcall", __func__, __FILE__, __LINE__);\
		}									\
	} while (0)

/*
 * Base of the block's dynamic shared memory region.
 *
 * CUDA documents no alignment guarantee for it -- the 256-byte guarantee in
 * the programming guide covers global memory only. In practice nvcc emits
 * .align 16 for extern __shared__ whatever the declared type, and clamps up
 * to 16 even when asked for less, so we ask for it explicitly rather than
 * lean on that. Kuiper.SHMem.c_shmems_inv assumes it; ./configure runs the
 * macro on the device and refuses to configure if it does not hold.
 */
#define KPR_SHMEM()									\
	({										\
		extern __shared__ __align__(16) char a[];				\
		(char*)a;								\
	})

#define KPR_SHMEM_AT(off)								\
	(void*)(KPR_SHMEM() + (off))

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
#define KRML_HOST_EPRINTF(s, ...)   fprintf(stderr, s, __VA_ARGS__)
#define KRML_HOST_EXIT(rc)          exit(rc)

static inline
void * __KPR_GPU_ALLOC(size_t sz, size_t len, const char * str, const char *func, const char *fname,
			     int line)
{
	void *ret = NULL;
	KRML_CHECK_SIZE(sz, len);
	__MUST(cudaMalloc(&ret, sz*len), str, func, fname, line);
	return ret;
}

#define KPR_GPU_ALLOC(sz, len)						\
	__KPR_GPU_ALLOC(sz, len, "KPR_GPU_ALLOC(" #sz ", " #len ")", __func__, __FILE__, __LINE__)

/* The rest of the device-memory runtime, for the Custard extraction (section
   79).  The Krml plugin emitted cudaFree/cudaMemcpy calls inline; the Custard
   rules go through these so that the call site is named in a MUST() failure
   the same way KPR_GPU_ALLOC's is. */

#define KPR_GPU_FREE(p)							\
	__MUST(cudaFree(p), "KPR_GPU_FREE(" #p ")", __func__, __FILE__, __LINE__)

#define KPR_SYNC_DEVICE()						\
	__MUST(cudaDeviceSynchronize(), "KPR_SYNC_DEVICE()", __func__,	\
	       __FILE__, __LINE__)

#define KPR_MEMCPY_H2D(dst, src, bytes)					\
	__MUST(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),	\
	       "KPR_MEMCPY_H2D(" #dst ", " #src ", " #bytes ")",		\
	       __func__, __FILE__, __LINE__)

#define KPR_MEMCPY_D2H(dst, src, bytes)					\
	__MUST(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),	\
	       "KPR_MEMCPY_D2H(" #dst ", " #src ", " #bytes ")",		\
	       __func__, __FILE__, __LINE__)

#define KPR_MEMCPY_D2D(dst, src, bytes)					\
	__MUST(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice),	\
	       "KPR_MEMCPY_D2D(" #dst ", " #src ", " #bytes ")",		\
	       __func__, __FILE__, __LINE__)

/*
 * Raise the dynamic shared memory cap for one kernel.
 *
 * cudaFuncSetAttribute is only needed above the 48KiB default; at or below it
 * the call does nothing, so the test keeps a CUDA runtime call out of the
 * common path.  [bytes] is a compile-time constant at every call site the
 * extraction produces, so nvcc folds the branch away entirely.
 *
 * The test is >= so a kernel sitting exactly on the default still opts in:
 * the call always succeeds at that size, which removes any doubt about the
 * boundary.
 */
#define KPR_SET_MAX_DYN_SHMEM(k, bytes)					\
	do {								\
		if ((size_t)(bytes) >= 49152) {				\
			__MUST(cudaFuncSetAttribute(			\
			         (const void *)(k),			\
			         cudaFuncAttributeMaxDynamicSharedMemorySize, \
			         (int)(bytes)),				\
			       "KPR_SET_MAX_DYN_SHMEM(" #k ", " #bytes ")", \
			       __func__, __FILE__, __LINE__);		\
		}							\
	} while (0)

static inline
void INFO ()
{
	printf("sizeof(short) = %zu\n", sizeof(short));
	printf("sizeof(int) = %zu\n", sizeof(int));
	printf("sizeof(long) = %zu\n", sizeof(long));
	printf("sizeof(long long) = %zu\n", sizeof(long long));

	printf("sizeof(unsigned short) = %zu\n", sizeof(unsigned short));
	printf("sizeof(unsigned int) = %zu\n", sizeof(unsigned int));
	printf("sizeof(unsigned long) = %zu\n", sizeof(unsigned long));
	printf("sizeof(unsigned long long) = %zu\n", sizeof(unsigned long long));
}

#define KPR_SHMEM_FITS(e) KPR_ASSERT((e) <= 101376) // 99KiB

/* FIXME: We should not emit KRML_CLITERAL, it happens when there are intermediate
tuples or structs that are not evaluated away. Ideally these values would not be
visible in the CUDA code at all. */
#define KRML_CLITERAL(a) (a)

static inline
cudaStream_t KPR_FRESH_STREAM() {
	cudaStream_t s;
	// Kuiper-created streams must be blocking because cudaMemcpy host -> device is assumed
	// to be blocking as far as the Kuiper semantics for it are concerned.
	// In reality, cudaMemcpy host -> device does not have to block, but it is an operation 
	// submitted to the NULL stream. Thus, only blocking streams are guaranteed to 
	// synchronize with it first.
	cudaError_t rc = cudaStreamCreate(&s);
	__MUST(rc, "cudaStreamCreate", __func__, __FILE__, __LINE__);\
	return s;
}

#endif /* KUIPER_H */

#define KPR_MUST_stream_destroy(s) MUST(cudaStreamDestroy(s))
#define KPR_MUST_stream_sync(s)    MUST(cudaStreamSynchronize(s))
