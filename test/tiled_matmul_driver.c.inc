#include "timing.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "matmul_common.c.inc"

#define DEFAULT_DIM   2048
#define DEFAULT_TILE  32
#define DEFAULT_LAPS  10

int main(int argc, char **argv)
{
    int i, j;
    int laps = DEFAULT_LAPS;
    size_t rows = DEFAULT_DIM;
    size_t shared = DEFAULT_DIM;
    size_t cols = DEFAULT_DIM;
    size_t tile = DEFAULT_TILE;
    bool check = true;

    progname = argv[0];

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

    et *m1_cpu, *m2_cpu, *m3_cpu;
    m1_cpu = (et *) malloc(rows * shared * sizeof(et));
    m2_cpu = (et *) malloc(shared * cols * sizeof(et));

    for (i = 0; i < rows; i++) {
        for (j = 0; j < shared; j++) {
            m1_cpu[i * shared + j] = 2 * i + j;
        }
    }
    for (i = 0; i < shared; i++) {
        for (j = 0; j < cols; j++) {
            m2_cpu[i * cols + j] = i + j;
        }
    }

    et *m1 = (et *) kpr_wait_alloc(sizeof(et), rows * shared);
    et *m2 = (et *) kpr_wait_alloc(sizeof(et), shared * cols);
    et *m3 = (et *) kpr_wait_alloc(sizeof(et), rows * cols);

    cudaMemcpy(m1, m1_cpu, rows * shared * sizeof(et), cudaMemcpyHostToDevice);
    cudaMemcpy(m2, m2_cpu, shared * cols * sizeof(et), cudaMemcpyHostToDevice);

    /* If we're gonna check, prepare the known-good result. */
    if (check)
        TIME_void(m3_cpu = cpu_mul(rows, shared, cols, m1_cpu, m2_cpu), NULL);

    for (int l = 0; l < laps; l++) {
        float t;
        fprintf(stderr, "Dynamic tile = %zu\n", tile);
        TIME_void(TOKENPASTE3(stem, et_lbl, _rrr)
                  (tile, rows, shared, cols, m1, m2, m3), &t);
        fprintf(stderr, "Estimated GFLOPS: %.3f\n",
                (rows * shared * cols * 2.0) / t / 1e9);
        if (check)
            cmp(rows, cols, m3, m3_cpu);
    }
    for (int l = 0; l < laps; l++) {
        float t;
        fprintf(stderr, "Static tile = 16\n");
        TIME_void(TOKENPASTE3(stem, et_lbl, _tile16_rrr)
                  (rows, shared, cols, m1, m2, m3), &t);
        fprintf(stderr, "Estimated GFLOPS: %.3f\n",
                (rows * shared * cols * 2.0) / t / 1e9);
        if (check)
            cmp(rows, cols, m3, m3_cpu);
    }
    for (int l = 0; l < laps; l++) {
        float t;
        fprintf(stderr, "Static tile = 32\n");
        TIME_void(TOKENPASTE3(stem, et_lbl, _tile32_rrr)
                  (rows, shared, cols, m1, m2, m3), &t);
        fprintf(stderr, "Estimated GFLOPS: %.3f\n",
                (rows * shared * cols * 2.0) / t / 1e9);
        if (check)
            cmp(rows, cols, m3, m3_cpu);
    }
    cudaFree(m3);

    printf("OK\n");

    return 0;
}
