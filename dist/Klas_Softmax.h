
#ifndef Klas_Softmax_H
#define Klas_Softmax_H

#include <kuiper.h>

void Klas_Softmax_softmax_f16(uint32_t lena, half * a);

void Klas_Softmax_softmax_f32(uint32_t lena, float *a);

void Klas_Softmax_softmax_f64(uint32_t lena, double *a);

#define Klas_Softmax_H_DEFINED
#endif                          /* Klas_Softmax_H */
