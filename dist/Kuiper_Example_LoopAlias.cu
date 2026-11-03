
#include "Kuiper_Example_LoopAlias.h"

void Kuiper_Example_LoopAlias_alias_step(uint32_t *p, uint32_t *q)
{
    bool again = true;
    while (again) {
        uint32_t old = *p;
        *q = 5U;
        again = false;
        *p = old + 1U;
    }
}
