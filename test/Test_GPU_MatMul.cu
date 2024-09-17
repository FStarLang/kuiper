#include "GPU_MatMul.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

#define N 1024

const size_t rows = N;
const size_t shared = N;
const size_t columns = N;

typedef uint64_t u64;

void pr(u64 *m) {
	int i,j ;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%lu ", m[i * 16+ j]);
		}
		printf("\n");
	}
}

u64 * naive_mul(u64 *m1, u64 *m2)
{
	u64 *m3 = (u64*)calloc(rows * columns, sizeof m3[0]);
	int i, j, k;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			u64 sum = 0;
			for (k = 0; k < shared; k++) {
				sum += m1[i * shared + k] * m2[k * columns + j];
			}
			m3[i * columns + j] = sum;
		}
	}
	return m3;
}

u64 * cpu_mul(u64 *m1, u64 *m2)
{
	u64 *m3 = (u64*)calloc(rows * columns, sizeof m3[0]);
	int i, j, k;
	for (i = 0; i < rows; i++) {
		for (k = 0; k < shared; k++) {
			for (j = 0; j < columns; j++) {
				m3[i * columns + j] += m1[i * shared + k] * m2[k * columns + j];
			}
		}
	}
	return m3;
}

int main()
{
	int i, j;

	u64 *m1, *m2;
	m1 = (u64*)malloc(rows * shared * sizeof m1[0]);
	m2 = (u64*)malloc(shared * columns * sizeof m2[0]);

	for (i = 0; i < rows; i++) {
		for (j = 0; j < shared; j++) {
			m1[i * shared + j] = 2*i+j;
		}
	}
	for (i = 0; i < shared; i++) {
		for (j = 0; j < columns; j++) {
			m2[i * columns + j] = i+j;
		}
	}
	
	// printf("M1\n"); pr(m1); printf("\n");
	// printf("M2\n"); pr(m2); printf("\n");

	u64 *m3 = NULL;
	for (int laps = 0; laps < 10; laps++) {
		free (m3);
		m3 = TIME(GPU_MatMul_main(rows, shared, columns, m1, m2), NULL);
	}

	u64 *m3_cpu = TIME(cpu_mul(m1, m2), NULL);

	// printf("M3\n"); pr(m3); printf("\n");
	// printf("M3 CPU\n"); pr(m3); printf("\n");
	
	// check
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			if (m3[i * columns + j] != m3_cpu[i * columns + j]) {
				printf("Error at %d %d: %lu != %lu\n", i, j, m3[i * columns + j], m3_cpu[i * columns + j]);
				return 1;
			}
		}
	}
	
	printf("OK\n");

	return 0;
}
