#include "Kuiper_MatMulTile_Async.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

typedef uint64_t u64;

u64 * cpu_mul(size_t rows, size_t shared, size_t columns, u64 *m1, u64 *m2)
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

int main(int argc, char **argv)
{
	int i, j;
	int laps = 5;
	size_t nn = 2048;
	size_t tile = 32;
	bool check = true;

	if (argc != 1 && argc != 5) {
		printf("Usage: %s [<laps> <nn> <tile> <check>]\n", argv[0]);
		return 1;
	} else if (argc == 5) {
		laps = atoi(argv[1]);
		nn = atoi(argv[2]);
		tile = atoi(argv[3]);
		check = atoi(argv[4]) != 0;
	}

	printf ("Laps = %d\n", laps);
	printf ("N = %lu\n", nn);
	printf ("Tile = %lu\n", tile);
	printf ("Check = %d\n", check);

	u64 *m1, *m2, *m3, *m4;
	m1 = (u64*)malloc(nn * nn * sizeof m1[0]);
	m2 = (u64*)malloc(nn * nn * sizeof m2[0]);
	m3 = (u64*)malloc(nn * nn * sizeof m2[0]);
	m4 = (u64*)malloc(nn * nn * sizeof m2[0]);

	for (i = 0; i < nn; i++) {
		for (j = 0; j < nn; j++) {
			m1[i * nn + j] = 2*i+j;
			m2[i * nn + j] = i+j;
			m3[i * nn + j] = 2*i+2*j;
			m4[i * nn + j] = i+2*j;
		}
	}

	u64 *mr = NULL;
	for (int l = 0; l < laps; l++) {
		float t;
		free (mr);
		mr = TIME(Kuiper_MatMulTile_Async_main(nn, tile, m1, m2, m3, m4), &t);
		// fprintf(stderr, "Estimated GIOPS: %.3f\n", (nn * nn * nn * 2.0) / t / 1e9);
	}

	if (check) {
		u64 *t1_cpu = TIME(cpu_mul(nn, nn, nn, m1, m2), NULL);
		u64 *t2_cpu = TIME(cpu_mul(nn, nn, nn, m3, m4), NULL);
		u64 *mr_cpu = TIME(cpu_mul(nn, nn, nn, t1_cpu, t2_cpu), NULL);
		for (i = 0; i < nn; i++) {
			for (j = 0; j < nn; j++) {
				if (mr[i * nn + j] != mr_cpu[i * nn + j]) {
					printf("Error at %d %d: %lu != %lu\n", i, j, mr[i * nn + j], mr_cpu[i * nn + j]);
					return 1;
				}
			}
		}
		free (t1_cpu);
		free (t2_cpu);
		free (mr_cpu);
	}

	printf("OK\n");
	return 0;
}
