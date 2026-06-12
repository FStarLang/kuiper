
#ifndef Klas_Reduce_H
#define Klas_Reduce_H

#include <kuiper.h>

void
Klas_Reduce_mean_fw_f32_row(uint32_t rows, uint32_t cols, float inv_cols,
                            float *x, float *y);

#define Klas_Reduce_H_DEFINED
#endif                          /* Klas_Reduce_H */
