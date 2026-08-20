
#include "Kuiper_Example_ArrayView_Test_EvenOdds3.h"

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds3_foo_even(uint32_t *a)
{
    return a[20U];
}

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds3_foo_odd(uint32_t *a)
{
    return a[21U];
}

uint32_t Kuiper_Example_ArrayView_Test_EvenOdds3_test(uint32_t *a)
{
    uint32_t *vr = a;
    uint32_t x = Kuiper_Example_ArrayView_Test_EvenOdds3_foo_even(a);
    return x + Kuiper_Example_ArrayView_Test_EvenOdds3_foo_odd(vr);
}

void Kuiper_Example_ArrayView_Test_EvenOdds3_test_write(uint32_t *a)
{
    uint32_t *vr = a;
    a[20U] = 42U;
    vr[41U] = 43U;
}
