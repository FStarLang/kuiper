#include "Kuiper_Example_OffsetMemcpy.h"
#include <stdint.h>
#include <stdio.h>

int main()
{
    uint64_t x = Kuiper_Example_OffsetMemcpy_main();
    printf("result = %lu\n", x);
    return 0;
}
