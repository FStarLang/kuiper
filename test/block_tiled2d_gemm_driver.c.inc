#include "test-common.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "matmul_common.c.inc"

#define DEFAULT_DIM   1024
// #define DEFAULT_TILE  32
#define DEFAULT_LAPS  2

int main(int argc, char **argv)
{
    int i, j;
    int laps = DEFAULT_LAPS;
    size_t rows = DEFAULT_DIM;
    size_t shared = DEFAULT_DIM;
    size_t cols = DEFAULT_DIM;
    bool check = true;

    progname = argv[0];

    if (argc == 6) {
        laps = atoi(argv[1]);
        rows = atoi(argv[2]);
        shared = atoi(argv[3]);
        cols = atoi(argv[4]);
        check = atoi(argv[5]);
    } else if (argc == 1) {
        /* use defaults */
    } else {
        fprintf(stderr,
                "Usage: %s [<laps> <rows> <shared> <columns> <check flag>]\n",
                argv[0]);
        return 1;
    }

    printf("+ Laps = %d\n", laps);
    printf("+ Rows = %lu\n", rows);
    printf("+ Shared = %lu\n", shared);
    printf("+ Columns = %lu\n", cols);

    et *m1_cpu, *m2_cpu, *m3_cpu;
    m1_cpu = (et *) malloc(rows * shared * sizeof(et));
    m2_cpu = (et *) malloc(shared * cols * sizeof(et));
    m3_cpu = (et *) malloc(rows * cols * sizeof(et));

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
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            m3_cpu[i * cols + j] = i - j;
        }
    }

    et *m1 = (et *) kpr_wait_alloc(sizeof(et), rows * shared);
    et *m2 = (et *) kpr_wait_alloc(sizeof(et), shared * cols);
    et *m3 = (et *) kpr_wait_alloc(sizeof(et), rows * cols);

    cudaMemcpy(m1, m1_cpu, rows * shared * sizeof(et), cudaMemcpyHostToDevice);
    cudaMemcpy(m2, m2_cpu, shared * cols * sizeof(et), cudaMemcpyHostToDevice);
    cudaMemcpy(m3, m3_cpu, rows * cols * sizeof(et), cudaMemcpyHostToDevice);

#define F TOKENPASTE5(stem, et_lbl, layouts, tile_sizes, regch_sizes)
#define STR(x) #x
#define XSTR(x) STR(x)

    et *gold = NULL;
    if (check) {
        // One extra lap if we're checking.
        TIME_void(gold =
                  cpu_sgemm(rows, shared, cols, m1_cpu, m2_cpu, m3_cpu,
                            GEMM_ALPHA, GEMM_BETA), NULL);

        F(PREARGS GEMM_ALPHA, GEMM_BETA, rows, shared, cols, m1, m2, m3);

        cmp(rows, cols, m3, gold);

        free(gold);
    }
    // Warmup
    for (int l = 0; l < laps / 10; l++) {
        F(PREARGS GEMM_ALPHA, GEMM_BETA, rows, shared, cols, m1, m2, m3);
    }

    // Now the actual bench
    float t = 0;
    for (int l = 0; l < laps; l++) {
        float delta;
        TIME_void(F
                  (PREARGS GEMM_ALPHA, GEMM_BETA, rows, shared, cols, m1, m2,
                   m3), &delta);
        t += delta;
    }
    uint64_t flops = laps * rows * shared * cols * 2.0;
    // I think the below is more accurate, but I'm reusing the
    // same formula from the sgemm.cu benchsuite
    // uint64_t flops = (rows * shared * cols * 2.0 + rows * cols * 3) / t;

    fprintf(stderr, ">> RES\t%s\t%.3f GFLOPS\n", XSTR(F),
            (double)flops / t / 1e9);

    cudaFree(m1);
    cudaFree(m2);
    cudaFree(m3);

    // printf("OK\n");

    return 0;
}
