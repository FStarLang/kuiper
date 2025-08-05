

#include "Kuiper_ArrayView_Test_EvenOdds2.h"

uint32_t Kuiper_ArrayView_Test_EvenOdds2_foo_even(uint32_t *a)
{
  return a[20U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds2_foo_odd(uint32_t *a)
{
  return a[21U];
}

void Kuiper_ArrayView_Test_EvenOdds2_write_even(uint32_t *a)
{
  a[20U] = 42U;
}

uint32_t Kuiper_ArrayView_Test_EvenOdds2_test_simpler(uint32_t *a)
{
  uint32_t *vr = a;
  uint32_t x = Kuiper_ArrayView_Test_EvenOdds2_foo_even(a);
  return x + Kuiper_ArrayView_Test_EvenOdds2_foo_odd(vr);
}

void Kuiper_ArrayView_Test_EvenOdds2_test_write(uint32_t *a)
{
  uint32_t *vr = a;
  a[20U] = 42U;
  vr[41U] = 43U;
}

