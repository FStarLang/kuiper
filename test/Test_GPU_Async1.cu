#include "GPU_Async1.h"
#include <stdint.h>
#include <stdio.h>

int main() {
	uint64_t x = GPU_Async1_main();
	printf("x = %lu\n", x);
	return 0;
}
