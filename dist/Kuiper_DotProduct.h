
#ifndef Kuiper_DotProduct_H
#define Kuiper_DotProduct_H

#include <kuiper.h>

float Kuiper_DotProduct_dotprod_f32(uint32_t lena, float *a1, float *a2);

double Kuiper_DotProduct_dotprod_f64(uint32_t lena, double *a1, double *a2);

uint32_t Kuiper_DotProduct_dotprod_u32(uint32_t lena, uint32_t * a1,
                                       uint32_t * a2);

uint64_t Kuiper_DotProduct_dotprod_u64(uint32_t lena, uint64_t * a1,
                                       uint64_t * a2);

#define Kuiper_DotProduct_H_DEFINED
#endif                          /* Kuiper_DotProduct_H */
