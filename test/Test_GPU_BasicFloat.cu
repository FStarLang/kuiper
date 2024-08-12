#include "GPU_BasicFloat.h"
#include <stdint.h>
#include <stdio.h>

int main() {
	float x = GPU_BasicFloat_main();
	printf("x = %f\n", x);
	return 0;
}
