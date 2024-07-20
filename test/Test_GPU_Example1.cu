#include "GPU_Example1.h"
#include <stdint.h>
#include <stdio.h>

int main() {
	uint64_t x = GPU_Example1_main();
	printf("x = %lu\n", x);
	return 0;
}
