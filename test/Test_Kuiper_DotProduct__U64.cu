#include <stdio.h>
#include <stdint.h>
#include "Kuiper_DotProduct.h"

int main()
{
    uint64_t a1[1024], a2[1024];
    int i;

    for (i = 0; i < 1024; i++)
        a1[i] = a2[i] = i;

    printf("%lu\n", Kuiper_DotProduct_dotprod_u64(1, a1, a2));
    printf("%lu\n", Kuiper_DotProduct_dotprod_u64(16, a1, a2));
    printf("%lu\n", Kuiper_DotProduct_dotprod_u64(128, a1, a2));
    printf("%lu\n", Kuiper_DotProduct_dotprod_u64(512, a1, a2));
    printf("%lu\n", Kuiper_DotProduct_dotprod_u64(1024, a1, a2));
    printf("%lu\n", Kuiper_DotProduct_dotprod_u64(1024, a1, a2));

    return 0;
}
