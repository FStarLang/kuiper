
#ifndef Klas_UGS_SwiGLU_H
#define Klas_UGS_SwiGLU_H

#include <kuiper.h>

void
Klas_UGS_SwiGLU_swiglu(uint32_t m,
                       uint32_t k,
                       uint32_t n,
                       half * gA,
                       half * gW, float *gPf32, half * gP, half * gOut);

#define Klas_UGS_SwiGLU_H_DEFINED
#endif                          /* Klas_UGS_SwiGLU_H */
