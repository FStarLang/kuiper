#include "Kuiper_Example_LoopAlias.h"
#include <stdio.h>

int main()
{
    uint32_t value = 0;
    Kuiper_Example_LoopAlias_alias_step(&value, &value);
    printf("value = %u\n", value);
    return value == 1 ? 0 : 1;
}
