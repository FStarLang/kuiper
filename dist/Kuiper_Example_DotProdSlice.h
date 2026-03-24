
#ifndef Kuiper_Example_DotProdSlice_H
#define Kuiper_Example_DotProdSlice_H

#include <kuiper.h>

float
Kuiper_Example_DotProdSlice_matmul_dotprod_via_slice_f32(uint32_t rows,
                                                         uint32_t shared,
                                                         uint32_t cols,
                                                         float *gA,
                                                         float *gB,
                                                         uint32_t i,
                                                         uint32_t j);

#define Kuiper_Example_DotProdSlice_H_DEFINED
#endif                          /* Kuiper_Example_DotProdSlice_H */
