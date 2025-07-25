

#include "Kuiper_ArrayView_Test1.h"

uint32_t Kuiper_ArrayView_Test1_test(uint32_t *a)
{
  return *(uint32_t *)a;
}

uint32_t Kuiper_ArrayView_Test1_test2(uint32_t *a)
{
  return ((uint32_t *)a)[49U];
}

void Kuiper_ArrayView_Test1_write1(uint32_t *a)
{
  *(uint32_t *)a = 123U;
}

void Kuiper_ArrayView_Test1_write2(uint32_t *a)
{
  ((uint32_t *)a)[49U] = 123U;
}

void Kuiper_ArrayView_Test1_write3(uint32_t *p)
{
  Kuiper_ArrayView_Test1_write2(p);
}

