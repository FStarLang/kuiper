/*
   nvcc test/driver.cu out/Kuiper.DotProduct2/Kuiper_DotProduct2.cu -I /home/guido/r/karamel/include/ -I /home/guido/r/karamel/krmllib/dist/minimal/ -I out/Kuiper.DotProduct2/
*/

#include <stdio.h>
#include <stdint.h>
#include "Kuiper_DotProduct2.h"

int main()
{
	uint64_t a1[1024], a2[1024];
	int i;

	for (i = 0; i < 1024; i++)
		a1[i] = a2[i] = i;

	uint64_t r = Kuiper_DotProduct2_main(a1, a2);

	printf("%lu\n", r);

	return 0;
}
