
#ifndef Klas_DotProduct_H
#define Klas_DotProduct_H

#include <kuiper.h>

float Klas_DotProduct_dotprod_f32(uint32_t lena, float *a1, float *a2);

double Klas_DotProduct_dotprod_f64(uint32_t lena, double *a1, double *a2);

uint32_t Klas_DotProduct_dotprod_u32(uint32_t lena, uint32_t * a1,
                                     uint32_t * a2);

uint64_t Klas_DotProduct_dotprod_u64(uint32_t lena, uint64_t * a1,
                                     uint64_t * a2);

#define Klas_DotProduct_H_DEFINED
#endif                          /* Klas_DotProduct_H */
