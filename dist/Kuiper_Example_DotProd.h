
#ifndef Kuiper_Example_DotProd_H
#define Kuiper_Example_DotProd_H

#include <kuiper.h>

/**
Compute the (i,j) element of the matrix product of A and B by extracting the
i-th row of A and the j-th column of B, then computing their dot product. The
resulting code looks just like the usual implementation.
*/
float
Kuiper_Example_DotProd_matmul_dotprod_via_slice_f32(uint32_t m,
                                                    uint32_t n,
                                                    uint32_t k,
                                                    float *gA,
                                                    float *gB,
                                                    uint32_t i, uint32_t j);

#define Kuiper_Example_DotProd_H_DEFINED
#endif                          /* Kuiper_Example_DotProd_H */
