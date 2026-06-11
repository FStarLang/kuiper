
#ifndef Kuiper_Example_CellRef_H
#define Kuiper_Example_CellRef_H

#include <kuiper.h>

uint32_t Kuiper_Example_CellRef_cell_get(uint32_t * a, uint32_t i);

void Kuiper_Example_CellRef_cell_set(uint32_t * a, uint32_t i, uint32_t w);

void Kuiper_Example_CellRef_array_set_via_ref(uint32_t * a, uint32_t j,
                                              uint32_t w);

#define Kuiper_Example_CellRef_H_DEFINED
#endif                          /* Kuiper_Example_CellRef_H */
