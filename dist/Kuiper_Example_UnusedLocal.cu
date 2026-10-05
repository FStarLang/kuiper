
#include "Kuiper_Example_UnusedLocal.h"

void Kuiper_Example_UnusedLocal_unused_zero(void)
{
    float buf = 0.0f;
    KRML_HOST_IGNORE(&buf);
}

void Kuiper_Example_UnusedLocal_unused_negative_infinity(void)
{
    float buf = 0.0f - INFINITY;
    KRML_HOST_IGNORE(&buf);
}
