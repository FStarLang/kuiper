#include "Kuiper_MatMul_Naive2.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

typedef uint64_t u64;

u64 *cpu_mul(size_t rows, size_t shared, size_t columns, u64 *m1, u64 *m2)
{
	u64 *m3 = (u64 *) calloc(rows * columns, sizeof m3[0]);
	int i, j, k;
	for (i = 0; i < rows; i++) {
		for (k = 0; k < shared; k++) {
			for (j = 0; j < columns; j++) {
				m3[i * columns + j] +=
				    m1[i * shared + k] * m2[k * columns + j];
			}
		}
	}
	return m3;
}

int main(int argc, char **argv)
{
	int i, j;
	int laps = 5;
	size_t rows = 1024;
	size_t shared = 1024;
	size_t columns = 1024;
	bool check = true;

	if (argc != 1 && argc != 6) {
		printf("Usage: %s [<laps> <rows> <shared> <columns> <check>]\n",
		       argv[0]);
		return 1;
	} else if (argc == 6) {
		laps = atoi(argv[1]);
		rows = atoi(argv[2]);
		shared = atoi(argv[3]);
		columns = atoi(argv[4]);
		check = atoi(argv[5]) != 0;
	}

	printf("Laps = %d\n", laps);
	printf("Rows = %lu\n", rows);
	printf("Shared = %lu\n", shared);
	printf("Columns = %lu\n", columns);
	printf("Check = %d\n", check);

	u64 *m1, *m2;
	m1 = (u64 *) malloc(rows * shared * sizeof m1[0]);
	m2 = (u64 *) malloc(shared * columns * sizeof m2[0]);

	for (i = 0; i < rows; i++) {
		for (j = 0; j < shared; j++) {
			m1[i * shared + j] = 2 * i + j;
		}
	}
	for (i = 0; i < shared; i++) {
		for (j = 0; j < columns; j++) {
			m2[i * columns + j] = i + j;
		}
	}

	u64 *m3 = NULL;
	for (int l = 0; l < laps; l++) {
		float t;
		free(m3);
		fprintf(stderr, "Standard\n");
		m3 = TIME(Kuiper_MatMul_Naive2_matmul_u64_rrr
			  (rows, shared, columns, m1, m2), &t);
		fprintf(stderr, "Estimated GIOPS: %.3f\n",
			(rows * shared * columns * 2.0) / t / 1e9);
	}

	if (check) {
		u64 *m3_cpu =
		    TIME(cpu_mul(rows, shared, columns, m1, m2), NULL);
		for (i = 0; i < rows; i++) {
			for (j = 0; j < columns; j++) {
				if (m3[i * columns + j] !=
				    m3_cpu[i * columns + j]) {
					printf("Error at %d %d: %lu != %lu\n",
					       i, j, m3[i * columns + j],
					       m3_cpu[i * columns + j]);
					return 1;
				}
			}
		}
	}

	for (int l = 0; l < laps; l++) {
		float t;
		free(m3);
		fprintf(stderr, "Flipped\n");
		/*
		   M1/M2 are row-major.

		   mult(R, M1, M2)
		   = TR(mult(R, TR(M2), TR(M1)))
		   = mult(C, TR(M2), TR(M1)
		   = mult(C, as_col_major M2, as_col_major M1)

		 */

		m3 = TIME(Kuiper_MatMul_Naive2_matmul_u64_ccc
			  (columns, shared, rows, m2, m1), &t);
		fprintf(stderr, "Estimated GIOPS: %.3f\n",
			(rows * shared * columns * 2.0) / t / 1e9);
	}

	if (check) {
		u64 *m3_cpu =
		    TIME(cpu_mul(rows, shared, columns, m1, m2), NULL);
		for (i = 0; i < rows; i++) {
			for (j = 0; j < columns; j++) {
				if (m3[i * columns + j] !=
				    m3_cpu[i * columns + j]) {
					printf("Error at %d %d: %lu != %lu\n",
					       i, j, m3[i * columns + j],
					       m3_cpu[i * columns + j]);
					return 1;
				}
			}
		}
	}

	printf("OK\n");

	return 0;
}
