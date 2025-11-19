
#ifndef Kuiper_Stencil_H
#define Kuiper_Stencil_H

#include <kuiper.h>

void
Kuiper_Stencil_stencil3x3_f32_add_rr(uint32_t rows, uint32_t cols, float *gIn,
                                     float *gOut);

void
Kuiper_Stencil_stencil3x3_i32_add_mul2_rc(uint32_t rows,
                                          uint32_t cols,
                                          uint32_t * gIn, uint32_t * gOut);

#define Kuiper_Stencil_H_DEFINED
#endif                          /* Kuiper_Stencil_H */
