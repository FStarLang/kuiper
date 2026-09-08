#include "Kuiper_Example_Async_Chain.h"
#include <stdint.h>
#include <stdio.h>

int main()
{
    uint64_t x = Kuiper_Example_Async_Chain_main();
    printf("x = %lu\n", x);
    return 0;
}
