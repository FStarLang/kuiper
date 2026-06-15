
#ifndef Klas_Misc_H
#define Klas_Misc_H

#include <kuiper.h>

void Klas_Misc_arange_i64(uint32_t len, uint64_t * out);

void
Klas_Misc_gather_bf16_u32(uint32_t lens,
                          uint32_t leni,
                          __nv_bfloat16 * src,
                          uint32_t * idx, __nv_bfloat16 * out);

#define Klas_Misc_H_DEFINED
#endif                          /* Klas_Misc_H */
