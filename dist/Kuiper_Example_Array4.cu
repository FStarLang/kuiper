
#include "Kuiper_Example_Array4.h"

void Kuiper_Example_Array4_test0(uint32_t *m)
{
    KRML_MAYBE_UNUSED_VAR(m);
}

uint32_t Kuiper_Example_Array4_test1(uint32_t *m)
{
    return m[1234U];
}

void Kuiper_Example_Array4_test2(uint32_t *m)
{
    m[1234U] = 42U;
}
