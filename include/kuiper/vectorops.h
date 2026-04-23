#ifndef KUIPER_VECTOROPS_H
#define KUIPER_VECTOROPS_H 1

#include <stdint.h>
#include <type_traits>

__device__
static inline
void vec_memcpy(void *dst, void *src)
{
    *((float4*)dst) = *((float4*)src);
}

#endif /* KUIPER_VECTOROPS_H */
