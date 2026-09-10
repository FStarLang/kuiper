
#ifndef Kuiper_Example_TensorCore_WGMMA_H
#define Kuiper_Example_TensorCore_WGMMA_H

#include <kuiper.h>

inline

    __device__ void
    Kuiper_Example_TensorCore_WGMMA_accumulate_twice(__nv_bfloat16 *a,
                                                     __nv_bfloat16 *b,
                                                     float *c);

inline

    __device__ void
    Kuiper_Example_TensorCore_WGMMA_multiply(__nv_bfloat16 *a, __nv_bfloat16 *b,
                                             float *c);

#define Kuiper_Example_TensorCore_WGMMA_H_DEFINED
#endif /* Kuiper_Example_TensorCore_WGMMA_H */
