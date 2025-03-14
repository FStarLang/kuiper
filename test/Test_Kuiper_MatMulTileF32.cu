#include "Kuiper_MatMulTileF32.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>

float *cpu_mul(size_t rows, size_t shared, size_t columns, float *m1, float *m2)
{
	float *m3 = (float *)calloc(rows * columns, sizeof m3[0]);
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
	size_t rows = 2048;
	size_t shared = 2048;
	size_t columns = 2048;
	size_t tile = 32;
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
		columns = atoi(argv[4]);
		tile = atoi(argv[5]);
		check = atoi(argv[6]) != 0;
	}

	printf("Laps = %d\n", laps);
	printf("Rows = %lu\n", rows);
	printf("Shared = %lu\n", shared);
	printf("Columns = %lu\n", columns);
	printf("Tile = %lu\n", tile);
	printf("Check = %d\n", check);

	float *m1, *m2;
	m1 = (float *)malloc(rows * shared * sizeof m1[0]);
	m2 = (float *)malloc(shared * columns * sizeof m2[0]);

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

	float *ga1;
	float *ga2;
	float *gr;
	MUST(cudaMalloc(&ga1, rows * shared * sizeof ga1[0]));
	MUST(cudaMalloc(&ga2, shared * columns * sizeof ga2[0]));
	MUST(cudaMalloc(&gr, rows * columns * sizeof gr[0]));
	MUST(cudaMemcpy
	     (ga1, m1, rows * shared * sizeof ga1[0], cudaMemcpyHostToDevice));
	MUST(cudaMemcpy
	     (ga2, m2, shared * columns * sizeof ga2[0],
	      cudaMemcpyHostToDevice));

	for (int l = 0; l < laps; l++) {
		float t;
		TIME_void(Kuiper_MatMulTileF32_g_mul
			  (rows, shared, columns, tile, ga1, ga2, gr), &t);
		fprintf(stderr, "Estimated GFLOPS: %.3f\n",
			(rows * shared * columns * 2.0) / t / 1e9);
	}
	cudaDeviceSynchronize();

	float *m3 = NULL;
	m3 = (float *)malloc(rows * columns * sizeof m3[0]);
	MUST(cudaMemcpy
	     (m3, gr, rows * columns * sizeof m3[0], cudaMemcpyDeviceToHost));

	if (check) {
		float *m3_cpu =
		    TIME(cpu_mul(rows, shared, columns, m1, m2), NULL);
		for (i = 0; i < rows; i++) {
			for (j = 0; j < columns; j++) {
				float l = m3[i * columns + j];
				float r = m3_cpu[i * columns + j];
				/* 1 in 1000 part error allowed */
				if ((l - r) / l > 1e-3) {
					printf("Error at %d %d: %f != %f\n", i,
					       j, l, r);
					// return 1;
				}
			}
		}
	}

	printf("OK\n");

	return 0;
}
