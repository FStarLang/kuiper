#ifndef __KUIPER__VECTOROPS_H
#define __KUIPER__VECTOROPS_H 1

#include <stdint.h>
#include <type_traits>

__device__
static inline
void vec_memcpy(void *dst, void *src)
{
    *((float4*)dst) = *((float4*)src);
}

#endif
