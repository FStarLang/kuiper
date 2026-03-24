
#include "Kuiper_Example_DotProdSlice.h"

float
Kuiper_Example_DotProdSlice_matmul_dotprod_via_slice_f32(uint32_t rows,
                                                         uint32_t shared,
                                                         uint32_t cols,
                                                         float *gA,
                                                         float *gB,
                                                         uint32_t i, uint32_t j)
{
    KRML_MAYBE_UNUSED_VAR(rows);
    uint32_t k = 0U;
    float sum = 0.0f;
    for (; k < shared; k++)
        sum += gA[i * shared + k] * gB[k * cols + j];
    return sum;
}
