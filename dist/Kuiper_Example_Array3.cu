
#include "Kuiper_Example_Array3.h"

void Kuiper_Example_Array3_test0(uint32_t *m)
{
    KRML_MAYBE_UNUSED_VAR(m);
}

uint32_t Kuiper_Example_Array3_test1(uint32_t *m)
{
    return m[123U];
}

void Kuiper_Example_Array3_test2(uint32_t *m)
{
    m[123U] = 42U;
}
