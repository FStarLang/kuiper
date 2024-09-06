#ifndef __PULSE__ATOMIC_H
#define __PULSE__ATOMIC_H 1

#include <stdint.h>

__device__
static inline
uint64_t atomic_add_u64(uint64_t *p, uint64_t v)
{
	return atomicAdd((unsigned long long*)p, v);
}

#endif
