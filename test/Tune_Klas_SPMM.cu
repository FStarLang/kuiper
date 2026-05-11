/* SPMM tuning driver.
 * Compiled with -Dstem=<kernel_func> by tuning/tune_spmm.sh.
 * Benchmarks a single SPMM kernel variant over given matrix dimensions. */

#include "spmm_common.c.inc"

const char *progname = "Tune_Klas_SPMM";

#define DEFAULT_LAPS   2
#define DEFAULT_DIM    4096
#define DEFAULT_DENSITY 10

#define STR(x) #x
#define XSTR(x) STR(x)

int main(int argc, char **argv)
{
    int laps = DEFAULT_LAPS;
    int rows = DEFAULT_DIM;
    int shared = DEFAULT_DIM;
    int cols = DEFAULT_DIM;
    int density_pct = DEFAULT_DENSITY;

    progname = argv[0];

    if (argc == 6) {
        laps = atoi(argv[1]);
        rows = atoi(argv[2]);
        shared = atoi(argv[3]);
        cols = atoi(argv[4]);
        density_pct = atoi(argv[5]);
    } else if (argc == 1) {
        /* use defaults */
    } else {
        fprintf(stderr, "Usage: %s [<laps> <rows> <shared> <cols> <density%%>]\n", argv[0]);
        return 1;
    }

    printf("+ Kernel = %s\n", XSTR(stem));
    printf("+ Laps = %d\n", laps);
    printf("+ Rows = %d\n", rows);
    printf("+ Shared = %d\n", shared);
    printf("+ Columns = %d\n", cols);
    printf("+ Density = %d%%\n", density_pct);

    /* Generate sparse A and dense B on host */
    float *AD = mk_dense_matrix_f32(rows, shared, density_pct);
    smatrix_f32_t A = sparsify_f32(AD, rows, shared);
    uint32_t *row_indices = mk_row_indices(rows, A);
    float *B = mk_dense_matrix_f32(shared, cols, 50);

    printf("+ NNZ = %u (%.1f%%)\n", A.nnz, 100.0 * A.nnz / ((double)rows * shared));

    /* Upload to device */
    smatrix_f32_t dA;
    uint32_t *drow_indices;
    float *dB, *dC;
    upload_spmm_f32(rows, shared, cols, A, row_indices, B, &dA, &drow_indices, &dB, &dC);

    /* Warmup */
    for (int l = 0; l < laps / 10 + 1; l++) {
        MUST(cudaMemset(dC, 0, sizeof(float) * rows * cols));
        stem(rows, shared, cols, dA, drow_indices, dB, dC);
    }

    /* Benchmark */
    float t = 0;
    for (int l = 0; l < laps; l++) {
        float delta;
        MUST(cudaMemset(dC, 0, sizeof(float) * rows * cols));
        TIME_void(stem(rows, shared, cols, dA, drow_indices, dB, dC), &delta);
        t += delta;
    }

    /* FLOPS: 2 * nnz * cols (each nonzero contributes a multiply+add per output column) */
    double flops = (double)laps * 2.0 * A.nnz * cols;
    fprintf(stderr, ">> RES\t%s\t%.3f GFLOPS\t(%.3f ms avg)\n",
            XSTR(stem), flops / t / 1e9, t / laps * 1000.0);

    free_spmm_device_f32(dA, drow_indices, dB, dC);
    free(AD);
    free(B);
    free(row_indices);
    free(A.elems);
    free(A.col_ind);
    free(A.row_off);

    return 0;
}
