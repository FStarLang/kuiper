#ifndef __KUIPER__VECTOROPS_H
#define __KUIPER__VECTOROPS_H 1

#include <stdint.h>
#include <type_traits>
#include <cuda/barrier>

__device__
static inline
void vec_memcpy(void *dst, void *src)
{
    *((float4*)dst) = *((float4*)src);
}

__device__
static inline
void vec_memcpy_async(void *dst, void *src, cuda::barrier<cuda::thread_scope::thread_scope_block>& barrier)
{
    memcpy_async(dst, src, cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), barrier);
}
#endif
