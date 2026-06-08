
#ifndef Kuiper_Kernel_FlashAttention_H
#define Kuiper_Kernel_FlashAttention_H

#include <kuiper.h>

void
Kuiper_Kernel_FlashAttention_flashattention_f32(float *gS,
                                                float *gKj,
                                                float *gVj,
                                                float *gQi,
                                                float *gOi,
                                                float *gl,
                                                float *gm, uint32_t tid);

#define Kuiper_Kernel_FlashAttention_H_DEFINED
#endif                          /* Kuiper_Kernel_FlashAttention_H */
