
#ifndef Kuiper_Example_Fused_GEMM_H
#define Kuiper_Example_Fused_GEMM_H

#include <kuiper.h>

/**
C <- sqrt(A) @ B, with A, B, C in fp16 and the accumulation done in fp32.
*/
void
Kuiper_Example_Fused_GEMM_gemm_sqrt_fused(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          half * gA, half * gB, half * gC);

#define Kuiper_Example_Fused_GEMM_H_DEFINED
#endif                          /* Kuiper_Example_Fused_GEMM_H */
