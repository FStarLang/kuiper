
#include "Kuiper_Example_ArrayView_Test1.h"

uint32_t Kuiper_Example_ArrayView_Test1_test(uint32_t *a)
{
    return a[FStar_Pervasives_coerce_eq((void *)0U, 0U)];
}

uint32_t Kuiper_Example_ArrayView_Test1_test2(uint32_t *a)
{
    return a[49U - FStar_Pervasives_coerce_eq((void *)0U, 0U)];
}

void Kuiper_Example_ArrayView_Test1_write1(uint32_t *a)
{
    a[FStar_Pervasives_coerce_eq((void *)0U, 0U)] = 123U;
}

void Kuiper_Example_ArrayView_Test1_write2(uint32_t *a)
{
    a[49U - FStar_Pervasives_coerce_eq((void *)0U, 0U)] = 123U;
}

void Kuiper_Example_ArrayView_Test1_write3(uint32_t *p)
{
    Kuiper_Example_ArrayView_Test1_write2(p);
}
