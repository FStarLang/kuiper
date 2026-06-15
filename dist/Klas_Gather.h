
#ifndef Klas_Gather_H
#define Klas_Gather_H

#include <kuiper.h>

void
Klas_Gather_gather_bf16_u64_2d(uint32_t cols,
                               uint32_t lensrc,
                               uint32_t lenout,
                               __nv_bfloat16 * src,
                               uint64_t * idx, __nv_bfloat16 * out);

#define Klas_Gather_H_DEFINED
#endif                          /* Klas_Gather_H */
