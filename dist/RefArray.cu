

#include "RefArray.h"

void RefArray_test(uint32_t **arr)
{
  uint32_t *r0 = arr[1U];
  *r0 += 1U;
}

void RefArray_use(void)
{
  KRML_CHECK_SIZE(sizeof (uint32_t *), (uint32_t)3U);
  uint32_t *arr[3U];
  for (uint32_t _i = 0U; _i < (uint32_t)3U; ++_i)
    arr[_i] = NULL;
  uint32_t buf = 1U;
  *arr = &buf;
  uint32_t buf0 = 2U;
  arr[1U] = &buf0;
  uint32_t buf1 = 3U;
  arr[2U] = &buf1;
  RefArray_test(arr);
}

