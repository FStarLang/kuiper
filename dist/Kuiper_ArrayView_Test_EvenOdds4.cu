

#include "Kuiper_ArrayView_Test_EvenOdds4.h"

uint32_t Kuiper_ArrayView_Test_EvenOdds4_x = 1U;

uint32_t Kuiper_ArrayView_Test_EvenOdds4_foo_even(uint32_t *a)
{
  return a[20U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds4_foo_odd(uint32_t *a)
{
  return a[21U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds4_foo_even_over_raw(uint32_t *a)
{
  return a[20U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds4_foo_odd_over_raw(uint32_t *a)
{
  return a[21U];
}

uint32_t Kuiper_ArrayView_Test_EvenOdds4_test_over_raw(uint32_t *a)
{
  uint32_t *a_ = a;
  return a_[20U] + a_[21U];
}

