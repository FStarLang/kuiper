#ifndef KUIPER_VECTOROPS_H
#define KUIPER_VECTOROPS_H 1

#include <stdint.h>
#include <type_traits>
#include <cuda_pipeline.h>

__device__
static inline
void vec_memcpy(void *dst, void *src)
{
    *((float4*)dst) = *((float4*)src);
}

// Only 16 byte variant for now
#define vec_memcpy_pipe(dst,src) __pipeline_memcpy_async((dst), (src), 16)

#endif /* KUIPER_VECTOROPS_H */
