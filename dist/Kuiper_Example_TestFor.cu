
#include "Kuiper_Example_TestFor.h"

void Kuiper_Example_TestFor_g(uint32_t x) { KRML_MAYBE_UNUSED_VAR(x); }

void Kuiper_Example_TestFor_test(void)
{
    uint32_t i_kpr0 = 0U;
    for (; i_kpr0 < 10U; i_kpr0++)
        Kuiper_Example_TestFor_g(i_kpr0);
}

void Kuiper_Example_TestFor_test_nested(void)
{
    uint32_t i_kpr2 = 0U;
    for (; i_kpr2 < 10U; i_kpr2++)
        Kuiper_Example_TestFor_test();
}

void Kuiper_Example_TestFor_test_nested_lit(void)
{
    uint32_t i_kpr6 = 0U;
    for (; i_kpr6 < 10U; i_kpr6++) {
        uint32_t i_kpr4 = 0U;
        for (; i_kpr4 < 20U; i_kpr4++)
            Kuiper_Example_TestFor_g(i_kpr4);
    }
}

void Kuiper_Example_TestFor_test_nested_lit_shadowed(void)
{
    uint32_t i_kpr10 = 0U;
    for (; i_kpr10 < 10U; i_kpr10++) {
        uint32_t j_kpr11 = i_kpr10;
        uint32_t i_kpr8 = 0U;
        for (; i_kpr8 < 20U; i_kpr8++)
            Kuiper_Example_TestFor_g(j_kpr11);
    }
}
