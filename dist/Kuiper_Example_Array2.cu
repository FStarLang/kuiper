
#include "Kuiper_Example_Array2.h"

void Kuiper_Example_Array2_test0(uint32_t *m)
{
    KRML_MAYBE_UNUSED_VAR(m);
}

uint32_t Kuiper_Example_Array2_test1(uint32_t *m)
{
    return m[12U];
}

void Kuiper_Example_Array2_test2(uint32_t *m)
{
    m[12U] = 42U;
}
