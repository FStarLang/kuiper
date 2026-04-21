
#include "Kuiper_Example_TestFor.h"

void Kuiper_Example_TestFor_g(uint32_t x)
{
    KRML_MAYBE_UNUSED_VAR(x);
}

void Kuiper_Example_TestFor_test(void)
{
    uint32_t i = 0U;
    for (; i < 10U; i++)
        Kuiper_Example_TestFor_g(i);
}

void Kuiper_Example_TestFor_test_nested(void)
{
    uint32_t i = 0U;
    for (; i < 10U; i++)
        Kuiper_Example_TestFor_test();
}

void Kuiper_Example_TestFor_test_nested_lit(void)
{
    uint32_t i = 0U;
    for (; i < 10U; i++) {
        uint32_t i = 0U;
        for (; i < 20U; i++)
            Kuiper_Example_TestFor_g(i);
    }
}

void Kuiper_Example_TestFor_test_nested_lit_shadowed(void)
{
    uint32_t i0 = 0U;
    for (; i0 < 10U; i0++) {
        uint32_t i = 0U;
        for (; i < 20U; i++)
            Kuiper_Example_TestFor_g(i0);
    }
}
