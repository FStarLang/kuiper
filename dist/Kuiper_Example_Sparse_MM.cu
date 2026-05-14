
#include "Kuiper_Example_Sparse_MM.h"

void
Kuiper_Example_Sparse_MM__mmsd_u32_rr(uint32_t rows,
                                      uint32_t shared,
                                      uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                                      uint32_t *gB, uint32_t *gC)
{
    KRML_MAYBE_UNUSED_VAR(shared);
    uint32_t i = 0U;
    for (; i < rows; i++) {
        uint32_t j = 0U;
        for (; j < cols; j++) {
            uint32_t dp = 0U;
            uint32_t k = gA.row_off[i];
            for (; k < gA.row_off[i + 1U]; k++)
                dp += gA.elems[k] * gB[gA.col_ind[k] * cols + j];
            gC[i * cols + j] = dp;
        }
    }
}

void
Kuiper_Example_Sparse_MM__mmsd_u32_cc(uint32_t rows,
                                      uint32_t shared,
                                      uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                                      uint32_t *gB, uint32_t *gC)
{
    uint32_t i = 0U;
    for (; i < rows; i++) {
        uint32_t j = 0U;
        for (; j < cols; j++) {
            uint32_t dp = 0U;
            uint32_t k = gA.row_off[i];
            for (; k < gA.row_off[i + 1U]; k++)
                dp += gA.elems[k] * gB[j * shared + gA.col_ind[k]];
            gC[j * rows + i] = dp;
        }
    }
}
