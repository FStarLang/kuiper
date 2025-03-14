#ifndef __PULSE__ATOMIC_H
#define __PULSE__ATOMIC_H 1

#include <stdint.h>
#include <type_traits>

__device__
static inline
uint32_t atomic_add_u32(uint32_t *p, uint32_t v)
{
    static_assert(std::is_same<unsigned int, uint32_t>::value, "xxx1");
	return atomicAdd((unsigned int*)p, v);
}

__device__
static inline
uint64_t atomic_add_u64(uint64_t *p, uint64_t v)
{
    static_assert(std::is_same<unsigned long, uint64_t>::value, "xxx2");
    // static_assert(std::is_same<unsigned long long, uint64_t>::value, "xxx3");
    // fails, why??
    static_assert(sizeof (unsigned long long) == 8,
      "unsigned long long must be uint64_t for this to be OK");

	return atomicAdd((unsigned long long *)p, v);
}

__device__
static inline
float atomic_add_f32(float *p, float v)
{
	return atomicAdd(p, v);
}

__device__
static inline
double atomic_add_f64(double* address, double val)
{
	// FIXME: We're lucky here that there is an implementation for arch<600,
	// but we should probably model GPU compute capabilities in the Pulse code.
#if __CUDA_ARCH__ >= 600
	return atomicAdd(p, v);
#else
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}

#endif
