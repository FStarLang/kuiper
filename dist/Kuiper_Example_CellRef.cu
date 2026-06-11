
#include "Kuiper_Example_CellRef.h"

uint32_t Kuiper_Example_CellRef_cell_get(uint32_t *a, uint32_t i)
{
    return *(a + i);
}

void Kuiper_Example_CellRef_cell_set(uint32_t *a, uint32_t i, uint32_t w)
{
    *(a + i) = w;
}

void Kuiper_Example_CellRef_array_set_via_ref(uint32_t *a, uint32_t j,
                                              uint32_t w)
{
    Kuiper_Example_CellRef_cell_set(a, j, w);
}
