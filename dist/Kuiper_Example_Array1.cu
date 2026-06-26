
#include "Kuiper_Example_Array1.h"

uint32_t Kuiper_Example_Array1_test1(uint32_t *m)
{
    return m[1U];
}

void Kuiper_Example_Array1_test2(uint32_t *m)
{
    m[1U] = 42U;
}
