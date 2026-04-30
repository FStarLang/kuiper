
#ifndef Klas_RowScale_H
#define Klas_RowScale_H

#include <kuiper.h>

void Klas_RowScale_rowscale_f16_rowmajor(uint32_t m, uint32_t n, half * a,
                                         half * b);

void Klas_RowScale_rowscale_f16_colmajor(uint32_t m, uint32_t n, half * a,
                                         half * b);

void Klas_RowScale_rowscale_f32_rowmajor(uint32_t m, uint32_t n, float *a,
                                         float *b);

void Klas_RowScale_rowscale_f32_colmajor(uint32_t m, uint32_t n, float *a,
                                         float *b);

void Klas_RowScale_rowscale_f64_rowmajor(uint32_t m, uint32_t n, double *a,
                                         double *b);

void Klas_RowScale_rowscale_f64_colmajor(uint32_t m, uint32_t n, double *a,
                                         double *b);

#define Klas_RowScale_H_DEFINED
#endif                          /* Klas_RowScale_H */
