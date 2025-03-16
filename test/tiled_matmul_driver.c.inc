#include "timing.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define DEFAULT_DIM   2048
#define DEFAULT_TILE  32
#define DEFAULT_LAPS  5

et *cpu_mul(size_t rows, size_t shared, size_t cols, et *m1, et *m2)
{
	et *m3 = (et *) calloc(rows * cols, sizeof m3[0]);
	int i, j, k;
	for (i = 0; i < rows; i++) {
		for (k = 0; k < shared; k++) {
			for (j = 0; j < cols; j++) {
				m3[i * cols + j] +=
				    m1[i * shared + k] * m2[k * cols + j];
			}
		}
	}
	return m3;
}

void cmp (size_t rows, size_t cols, et *m1,  et* m2) {
	int i, j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			bool ok;
			#ifdef EXACT
			ok = m1[i * cols + j] == m2[i * cols + j];
			#else
			ok = ((m1[i * cols + j] - m2[i * cols + j]) / (double)m2[i * cols + j]) <= TOLERANCE;
			#endif
			if (!ok) {
				printf("Error at pos (%d, %d): %" PRIet " != %" PRIet "\n",
				       i, j, m1[i * cols + j],
				       m2[i * cols + j]);
				exit(1);
			}
		}
	}
}

int main(int argc, char **argv)
{
	int i, j;
	int laps = DEFAULT_LAPS;
	size_t rows = DEFAULT_DIM;
	size_t shared = DEFAULT_DIM;
	size_t cols = DEFAULT_DIM;
	size_t tile = DEFAULT_TILE;
	bool check = true;

	if (argc != 1 && argc != 7) {
		printf
		    ("Usage: %s [<laps> <rows> <shared> <columns> <tile> <check>]\n",
		     argv[0]);
		return 1;
	} else if (argc == 7) {
		laps = atoi(argv[1]);
		rows = atoi(argv[2]);
		shared = atoi(argv[3]);
		cols = atoi(argv[4]);
		tile = atoi(argv[5]);
		check = atoi(argv[6]) != 0;
	}

	printf("Laps = %d\n", laps);
	printf("Rows = %lu\n", rows);
	printf("Shared = %lu\n", shared);
	printf("Columns = %lu\n", cols);
	printf("Tile = %lu\n", tile);
	printf("Check = %d\n", check);

	et *m1, *m2, *m3_cpu;
	m1 = (et *) malloc(rows * shared * sizeof m1[0]);
	m2 = (et *) malloc(shared * cols * sizeof m2[0]);

	for (i = 0; i < rows; i++) {
		for (j = 0; j < shared; j++) {
			m1[i * shared + j] = 2 * i + j;
		}
	}
	for (i = 0; i < shared; i++) {
		for (j = 0; j < cols; j++) {
			m2[i * cols + j] = i + j;
		}
	}

	if (check)
		m3_cpu = cpu_mul(rows, shared, cols, m1, m2);

	et *m3 = NULL;
	for (int l = 0; l < laps; l++) {
		float t;
		free(m3);
		fprintf(stderr, "Static tile = 16\n");
		m3 = TIME(TOKENPASTE3(stem, et_lbl, _tile16_rrr)(rows, shared, cols, m1, m2), &t);
		fprintf(stderr, "Estimated GFLOPS: %.3f\n",
			(rows * shared * cols * 2.0) / t / 1e9);
		if (check) cmp (rows, cols, m3, m3_cpu);
	}
	for (int l = 0; l < laps; l++) {
		float t;
		free(m3);
		fprintf(stderr, "Static tile = 32\n");
		m3 = TIME(TOKENPASTE3(stem, et_lbl, _tile32_rrr)(rows, shared, cols, m1, m2), &t);
		fprintf(stderr, "Estimated GFLOPS: %.3f\n",
			(rows * shared * cols * 2.0) / t / 1e9);
		if (check) cmp (rows, cols, m3, m3_cpu);
	}
	for (int l = 0; l < laps; l++) {
		float t;
		free(m3);
		fprintf(stderr, "Dynamic tile = %d\n", tile);
		m3 = TIME(TOKENPASTE3(stem, et_lbl, _rrr)(tile, rows, shared, cols, m1, m2), &t);
		fprintf(stderr, "Estimated GFLOPS: %.3f\n",
			(rows * shared * cols * 2.0) / t / 1e9);
		if (check) cmp (rows, cols, m3, m3_cpu);
	}


	printf("OK\n");

	return 0;
}
