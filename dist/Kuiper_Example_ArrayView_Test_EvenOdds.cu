
#include "Kuiper_Example_ArrayView_Test_EvenOdds.h"

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds_foo_even(uint32_t *a)
{
    return a[FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U];
}

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds_foo_odd(uint32_t *a)
{
    return a[1U + FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U];
}

void Kuiper_Example_ArrayView_Test_EvenOdds_foo_odd_modify(uint32_t *a)
{
    a[1U + FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U] = 42U;
}
