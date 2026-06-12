
#include "Klas_Elementwise.h"

__global__
/**
  hoisted when extracting silu_fw_bf16
*/
static void __hoisted_silu_fw_bf16_0(uint32_t lena, __nv_bfloat16 *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        __nv_bfloat16 x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            __hmul(x,
                   __hdiv(CUDART_ONE_BF16,
                          __hadd(CUDART_ONE_BF16,
                                 kpr_bf16exp(__hsub(CUDART_ZERO_BF16, x)))));
    }
}

void Klas_Elementwise_silu_fw_bf16(uint32_t lena, __nv_bfloat16 *a)
{
    KPR_KCALL(__hoisted_silu_fw_bf16_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting neg_fw_bf16
*/
static void __hoisted_neg_fw_bf16_0(uint32_t lena, __nv_bfloat16 *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            __hsub(CUDART_ZERO_BF16, a[1024U * blockIdx.x + threadIdx.x]);
}

void Klas_Elementwise_neg_fw_bf16(uint32_t lena, __nv_bfloat16 *a)
{
    KPR_KCALL(__hoisted_neg_fw_bf16_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting rsqrt_fw_f32
*/
static void __hoisted_rsqrt_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            rsqrtf(a[1024U * blockIdx.x + threadIdx.x]);
}

void Klas_Elementwise_rsqrt_fw_f32(uint32_t lena, float *a)
{
    KPR_KCALL(__hoisted_rsqrt_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting square_fw_f32
*/
static void __hoisted_square_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = x * x;
    }
}

void Klas_Elementwise_square_fw_f32(uint32_t lena, float *a)
{
    KPR_KCALL(__hoisted_square_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting cos_fw_f32
*/
static void __hoisted_cos_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            cosf(a[1024U * blockIdx.x + threadIdx.x]);
}

void Klas_Elementwise_cos_fw_f32(uint32_t lena, float *a)
{
    KPR_KCALL(__hoisted_cos_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting sin_fw_f32
*/
static void __hoisted_sin_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            sinf(a[1024U * blockIdx.x + threadIdx.x]);
}

void Klas_Elementwise_sin_fw_f32(uint32_t lena, float *a)
{
    KPR_KCALL(__hoisted_sin_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting add_fw_bf16
*/
static void __hoisted_add_fw_bf16_0(uint32_t lena, __nv_bfloat16 *a,
                                    __nv_bfloat16 *b)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            __hadd(a[1024U * blockIdx.x + threadIdx.x],
                   b[1024U * blockIdx.x + threadIdx.x]);
}

void Klas_Elementwise_add_fw_bf16(uint32_t lena, __nv_bfloat16 *a,
                                  __nv_bfloat16 *b)
{
    KPR_KCALL(__hoisted_add_fw_bf16_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, b);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting mul_fw_bf16
*/
static void __hoisted_mul_fw_bf16_0(uint32_t lena, __nv_bfloat16 *a,
                                    __nv_bfloat16 *b)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            __hmul(a[1024U * blockIdx.x + threadIdx.x],
                   b[1024U * blockIdx.x + threadIdx.x]);
}

void Klas_Elementwise_mul_fw_bf16(uint32_t lena, __nv_bfloat16 *a,
                                  __nv_bfloat16 *b)
{
    KPR_KCALL(__hoisted_mul_fw_bf16_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, b);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting mul_fw_f32
*/
static void __hoisted_mul_fw_f32_0(uint32_t lena, float *a, float *b)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *=
            b[1024U * blockIdx.x + threadIdx.x];
}

void Klas_Elementwise_mul_fw_f32(uint32_t lena, float *a, float *b)
{
    KPR_KCALL(__hoisted_mul_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, b);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting add_const_fw_f32
*/
static void __hoisted_add_const_fw_f32_0(float c, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] += c;
}

void Klas_Elementwise_add_const_fw_f32(float c, uint32_t lena, float *a)
{
    KPR_KCALL(__hoisted_add_const_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, c, lena, a);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting mul_const_fw_f32
*/
static void __hoisted_mul_const_fw_f32_0(float c, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

void Klas_Elementwise_mul_const_fw_f32(float c, uint32_t lena, float *a)
{
    KPR_KCALL(__hoisted_mul_const_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, c, lena, a);
    MUST(cudaDeviceSynchronize());
}
