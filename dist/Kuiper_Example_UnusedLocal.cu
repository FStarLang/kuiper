
#include "Kuiper_Example_UnusedLocal.h"

void Kuiper_Example_UnusedLocal_unused_zero(void)
{
    float buf = (float) 0LL;
    KRML_HOST_IGNORE(&buf);
}

void Kuiper_Example_UnusedLocal_unused_negative_infinity(void)
{
    float buf = (float) 0LL - INFINITY;
    KRML_HOST_IGNORE(&buf);
}
