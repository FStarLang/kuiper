

#ifndef __Kuiper_DotProduct_H
#define __Kuiper_DotProduct_H


#include <kuiper.h>

float_t Kuiper_DotProduct_dotprod_f32(size_t lena, float_t *a1, float_t *a2);

double_t Kuiper_DotProduct_dotprod_f64(size_t lena, double_t *a1, double_t *a2);

uint32_t Kuiper_DotProduct_dotprod_u32(size_t lena, uint32_t *a1, uint32_t *a2);

uint64_t Kuiper_DotProduct_dotprod_u64(size_t lena, uint64_t *a1, uint64_t *a2);


#define __Kuiper_DotProduct_H_DEFINED
#endif
