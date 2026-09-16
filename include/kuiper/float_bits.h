#ifndef KUIPER_FLOAT_BITS_H
#define KUIPER_FLOAT_BITS_H

#include <string.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/* Compare representations, preserving zero signs and NaN payloads.
 * Host memcmp avoids aliasing violations; device bit casts emit no conversion.
 */
static __host__ __device__ __forceinline__ bool kpr_f32_bit_eq(float x, float y)
{
#ifdef __CUDA_ARCH__
    return __float_as_uint(x) == __float_as_uint(y);
#else
    return memcmp(&x, &y, sizeof x) == 0;
#endif
}

static __host__ __device__ __forceinline__ bool kpr_f64_bit_eq(
    double x, double y)
{
#ifdef __CUDA_ARCH__
    return __double_as_longlong(x) == __double_as_longlong(y);
#else
    return memcmp(&x, &y, sizeof x) == 0;
#endif
}

static __host__ __device__ __forceinline__ bool kpr_f16_bit_eq(
    __half x, __half y)
{
    return __half_as_ushort(x) == __half_as_ushort(y);
}

static __host__ __device__ __forceinline__ bool kpr_bf16_bit_eq(
    __nv_bfloat16 x, __nv_bfloat16 y)
{
    return __bfloat16_as_ushort(x) == __bfloat16_as_ushort(y);
}

#endif
