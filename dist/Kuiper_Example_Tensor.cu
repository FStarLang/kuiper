
#include "Kuiper_Example_Tensor.h"

void Kuiper_Example_Tensor_test0(uint32_t *m)
{
    KRML_MAYBE_UNUSED_VAR(m);
}

uint32_t Kuiper_Example_Tensor_test1(uint32_t *m)
{
    return m[22U];
}

void Kuiper_Example_Tensor_test2(uint32_t *m)
{
    m[22U] = 42U;
}
