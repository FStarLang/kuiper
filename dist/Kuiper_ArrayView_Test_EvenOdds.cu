

#include "Kuiper_ArrayView_Test_EvenOdds.h"

uint32_t Kuiper_ArrayView_Test_EvenOdds_foo_even(uint32_t *a)
{
  return ((uint32_t *)a)[20U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds_foo_odd(uint32_t *a)
{
  return ((uint32_t *)a)[21U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds_test(uint32_t *a)
{
  uint32_t *va3 = a;
  uint32_t *vr = va3;
  uint32_t x = Kuiper_ArrayView_Test_EvenOdds_foo_even(va3);
  return x + Kuiper_ArrayView_Test_EvenOdds_foo_odd(vr);
}

uint32_t Kuiper_ArrayView_Test_EvenOdds_test_simpler(uint32_t *a)
{
  uint32_t *vr = a;
  uint32_t x = Kuiper_ArrayView_Test_EvenOdds_foo_even(a);
  return x + Kuiper_ArrayView_Test_EvenOdds_foo_odd(vr);
}

