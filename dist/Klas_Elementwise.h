
#ifndef Klas_Elementwise_H
#define Klas_Elementwise_H

#include <kuiper.h>

void Klas_Elementwise_silu_fw_bf16(uint32_t lena, __nv_bfloat16 * a);

void Klas_Elementwise_neg_fw_bf16(uint32_t lena, __nv_bfloat16 * a);

void Klas_Elementwise_rsqrt_fw_f32(uint32_t lena, float *a);

void Klas_Elementwise_square_fw_f32(uint32_t lena, float *a);

void Klas_Elementwise_cos_fw_f32(uint32_t lena, float *a);

void Klas_Elementwise_sin_fw_f32(uint32_t lena, float *a);

void Klas_Elementwise_add_fw_bf16(uint32_t lena, __nv_bfloat16 * a,
                                  __nv_bfloat16 * b);

void Klas_Elementwise_mul_fw_bf16(uint32_t lena, __nv_bfloat16 * a,
                                  __nv_bfloat16 * b);

void Klas_Elementwise_mul_fw_f32(uint32_t lena, float *a, float *b);

void Klas_Elementwise_add_const_fw_f32(float c, uint32_t lena, float *a);

void Klas_Elementwise_mul_const_fw_f32(float c, uint32_t lena, float *a);

#define Klas_Elementwise_H_DEFINED
#endif                          /* Klas_Elementwise_H */
