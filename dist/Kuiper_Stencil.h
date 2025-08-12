

#ifndef Kuiper_Stencil_H
#define Kuiper_Stencil_H

#include <kuiper.h>

void
Kuiper_Stencil_stencil3x3_f32_add_rr(size_t rows, size_t cols, float_t *gIn, float_t *gOut);

void
Kuiper_Stencil_stencil3x3_i32_add_mul2_rc(
  size_t rows,
  size_t cols,
  uint32_t *gIn,
  uint32_t *gOut
);


#define Kuiper_Stencil_H_DEFINED
#endif /* Kuiper_Stencil_H */
