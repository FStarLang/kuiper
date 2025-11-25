
#include "TestFor.h"

void TestFor_g(uint32_t x)
{
    KRML_MAYBE_UNUSED_VAR(x);
}

void TestFor_test(void)
{
    uint32_t i = 0U;
    for (; i < 10U; i++)
        TestFor_g(i);
}

void TestFor_test_nested(void)
{
    uint32_t i = 0U;
    for (; i < 10U; i++)
        TestFor_test();
}

void TestFor_test_nested_lit(void)
{
    uint32_t i = 0U;
    for (; i < 10U; i++) {
        uint32_t i = 0U;
        for (; i < 20U; i++)
            TestFor_g(i);
    }
}

void TestFor_test_nested_lit_shadowed(void)
{
    uint32_t i0 = 0U;
    for (; i0 < 10U; i0++) {
        uint32_t i = 0U;
        for (; i < 20U; i++)
            TestFor_g(i0);
    }
}
