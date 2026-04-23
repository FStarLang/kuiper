
#ifndef Klas_LogSoftmax_H
#define Klas_LogSoftmax_H

#include <kuiper.h>

void Klas_LogSoftmax_log_softmax_n_f16(uint32_t nth, uint32_t lena, half * a);

void Klas_LogSoftmax_log_softmax_n_f32(uint32_t nth, uint32_t lena, float *a);

void Klas_LogSoftmax_log_softmax_n_f64(uint32_t nth, uint32_t lena, double *a);

void Klas_LogSoftmax_log_softmax_f16(uint32_t lena, half * a);

void Klas_LogSoftmax_log_softmax_f32(uint32_t lena, float *a);

void Klas_LogSoftmax_log_softmax_f64(uint32_t lena, double *a);

#define Klas_LogSoftmax_H_DEFINED
#endif                          /* Klas_LogSoftmax_H */
