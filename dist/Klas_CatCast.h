
#ifndef Klas_CatCast_H
#define Klas_CatCast_H

#include <kuiper.h>

__nv_bfloat16
    * Klas_CatCast_cat2_bf16(uint32_t lena, uint32_t lenb, __nv_bfloat16 * a,
                             __nv_bfloat16 * b);

float *Klas_CatCast_cast_bf16_to_f32(uint32_t len, __nv_bfloat16 * a);

__nv_bfloat16 *Klas_CatCast_cast_f32_to_bf16(uint32_t len, float *a);

__nv_bfloat16 *Klas_CatCast_cast_bf16_to_bf16(uint32_t len, __nv_bfloat16 * a);

#define Klas_CatCast_H_DEFINED
#endif                          /* Klas_CatCast_H */
