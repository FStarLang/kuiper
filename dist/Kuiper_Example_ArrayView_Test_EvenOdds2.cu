
#include "Kuiper_Example_ArrayView_Test_EvenOdds2.h"

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds2_foo_even(uint32_t *a)
{
    return a[FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U];
}

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds2_foo_odd(uint32_t *a)
{
    return a[1U + FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U];
}

void Kuiper_Example_ArrayView_Test_EvenOdds2_write_even(uint32_t *a)
{
    a[FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U] = 42U;
}

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds2_test_simpler(uint32_t *a)
{
    uint32_t *vr = a;
    uint32_t x = Kuiper_Example_ArrayView_Test_EvenOdds2_foo_even(a);
    return x + Kuiper_Example_ArrayView_Test_EvenOdds2_foo_odd(vr);
}

void Kuiper_Example_ArrayView_Test_EvenOdds2_test_write(uint32_t *a)
{
    uint32_t *vl = a;
    uint32_t *vr = a;
    vl[FStar_Pervasives_coerce_eq((void *)0U, 10U) * 2U] = 42U;
    vr[1U + FStar_Pervasives_coerce_eq((void *)0U, 20U) * 2U] = 43U;
}
