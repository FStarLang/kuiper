
#include "Klas_CatCast.h"

__nv_bfloat16 *Klas_CatCast_cat2_bf16(uint32_t len, __nv_bfloat16 *a,
                                      __nv_bfloat16 *b)
{
    KRML_MAYBE_UNUSED_VAR(a);
    KRML_MAYBE_UNUSED_VAR(b);
    return (__nv_bfloat16 *) KPR_GPU_ALLOC(sizeof(__nv_bfloat16), len + len);
}

__global__
/**
  hoisted when extracting cast_bf16_to_f32
*/
static void __hoisted_cast_bf16_to_f32_0(uint32_t len, __nv_bfloat16 *a,
                                         float *c)
{
    if (1024U * blockIdx.x + threadIdx.x < len)
        c[1024U * blockIdx.x + threadIdx.x] =
            __bfloat162float(a[1024U * blockIdx.x + threadIdx.x]);
}

float *Klas_CatCast_cast_bf16_to_f32(uint32_t len, __nv_bfloat16 *a)
{
    float *c = (float *)KPR_GPU_ALLOC(sizeof(float), len);
    KPR_KCALL(__hoisted_cast_bf16_to_f32_0,
              len / 1024U + (uint32_t) (len % 1024U != 0U),
              1024U, 0U, len, a, c);
    MUST(cudaDeviceSynchronize());
    return c;
}

__global__
/**
  hoisted when extracting cast_f32_to_bf16
*/
static void __hoisted_cast_f32_to_bf16_0(uint32_t len, float *a,
                                         __nv_bfloat16 *c)
{
    if (1024U * blockIdx.x + threadIdx.x < len)
        c[1024U * blockIdx.x + threadIdx.x] =
            __float2bfloat16(a[1024U * blockIdx.x + threadIdx.x]);
}

__nv_bfloat16 *Klas_CatCast_cast_f32_to_bf16(uint32_t len, float *a)
{
    __nv_bfloat16 *c =
        (__nv_bfloat16 *) KPR_GPU_ALLOC(sizeof(__nv_bfloat16), len);
    KPR_KCALL(__hoisted_cast_f32_to_bf16_0,
              len / 1024U + (uint32_t) (len % 1024U != 0U), 1024U, 0U, len, a,
              c);
    MUST(cudaDeviceSynchronize());
    return c;
}

__global__
/**
  hoisted when extracting cast_bf16_to_bf16
*/
static void __hoisted_cast_bf16_to_bf16_0(uint32_t len, __nv_bfloat16 *a,
                                          __nv_bfloat16 *c)
{
    if (1024U * blockIdx.x + threadIdx.x < len)
        c[1024U * blockIdx.x + threadIdx.x] =
            a[1024U * blockIdx.x + threadIdx.x];
}

__nv_bfloat16 *Klas_CatCast_cast_bf16_to_bf16(uint32_t len, __nv_bfloat16 *a)
{
    __nv_bfloat16 *c =
        (__nv_bfloat16 *) KPR_GPU_ALLOC(sizeof(__nv_bfloat16), len);
    KPR_KCALL(__hoisted_cast_bf16_to_bf16_0,
              len / 1024U + (uint32_t) (len % 1024U != 0U), 1024U, 0U, len, a,
              c);
    MUST(cudaDeviceSynchronize());
    return c;
}
