
#include "Kuiper_ArrayView_Test_EvenOdds.h"

uint32_t Kuiper_ArrayView_Test_EvenOdds_foo_even(uint32_t *a)
{
    return a[20U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds_foo_odd(uint32_t *a)
{
    return a[21U];
}

void Kuiper_ArrayView_Test_EvenOdds_foo_odd_modify(uint32_t *a)
{
    a[21U] = 42U;
}
