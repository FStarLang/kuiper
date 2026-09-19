#include "spmm_common.c.inc"

const char *progname = __FILE__;

typedef Kuiper_Sparse_Matrix_smatrix__float smatrix_t;
typedef decltype(&Klas_SPMM_spmm_f32) spmm_fn;

static int g_ok = 1;
static int g_tests = 0;
static bool do_check = 1;

static void cpu_matmul(
    float *A, float *B, float *C, int rows, int shared, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            float sum = 0;
            for (int k = 0; k < shared; k++)
                sum += A[i * shared + k] * B[k * cols + j];
            C[i * cols + j] = sum;
        }
}

static void run_spmm(const char *label, float *AD, int rows, int shared,
    int cols, spmm_fn spmm = Klas_SPMM_spmm_f32)
{
    smatrix_t A = sparsify_f32(AD, rows, shared);
    uint32_t *row_indices = mk_row_indices(rows, A);
    float *B = mk_dense_matrix_f32(shared, cols, 50);
    float *CD = (float *) calloc(rows * cols, sizeof CD[0]);

    if (do_check)
        cpu_matmul(AD, B, CD, rows, shared, cols);

    smatrix_t dA;
    uint32_t *drow_indices;
    float *dB, *dC;
    upload_spmm_f32(
        rows, shared, cols, A, row_indices, B, &dA, &drow_indices, &dB, &dC);

    float t;
    TIME_void(spmm(rows, shared, cols, dA, drow_indices, dB, dC), &t);
    fprintf(stderr,
        ">>> RES (rows=%d, shared=%d, cols=%d, sparsity=%.2f%%) \t GFLOPS: "
        "%.3f\n",
        rows, shared, cols, (1.0 - (double) A.nnz / (rows * shared)) * 100.0,
        (A.nnz * cols * 2.0) / t / 1e9);

    float *C = (float *) calloc(rows * cols, sizeof C[0]);
    MUST(cudaMemcpy(C, dC, sizeof C[0] * rows * cols, cudaMemcpyDeviceToHost));

    free_spmm_device_f32(dA, drow_indices, dB, dC);

    g_tests++;
    int mismatches = 0;
    if (do_check) {
        const float atol = 1e-3;
        const float rtol = 1e-2;
        for (int i = 0; i < rows * cols; i++) {
            if (fabs(C[i] - CD[i]) > atol + rtol * fabs(CD[i])) {
                if (mismatches == 0)
                    fprintf(stderr,
                        "FAIL %s: first mismatch at (%d,%d): "
                        "got %f, ref %f\n",
                        label, i / cols, i % cols, C[i], CD[i]);
                mismatches++;
            }
        }
        if (mismatches > 0) {
            fprintf(stderr, "FAIL %s: %d mismatches out of %d\n", label,
                mismatches, rows * cols);
            g_ok = 0;
        }
    }

    free(row_indices);
    free(B);
    free(C);
    free(CD);
    free(A.elems);
    free(A.col_ind);
    free(A.row_off);
}

static void test_random(int rows, int shared, int cols, int density_pct)
{
    char label[128];
    snprintf(label, sizeof label, "random(%dx%dx%d, %d%%)", rows, shared, cols,
        density_pct);
    float *AD = mk_dense_matrix_f32(rows, shared, density_pct);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

static void test_identity(int n, int cols)
{
    char label[128];
    snprintf(label, sizeof label, "identity(%dx%d, cols=%d)", n, n, cols);
    float *AD = mk_identity_matrix_f32(n, n);
    run_spmm(label, AD, n, n, cols);
    free(AD);
}

static void test_empty(int rows, int shared, int cols)
{
    char label[128];
    snprintf(label, sizeof label, "empty(%dx%dx%d)", rows, shared, cols);
    float *AD = (float *) calloc(rows * shared, sizeof AD[0]);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

static void test_single_per_row(int rows, int shared, int cols)
{
    char label[128];
    snprintf(
        label, sizeof label, "single_per_row(%dx%dx%d)", rows, shared, cols);
    float *AD = mk_single_per_row_f32(rows, shared);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

static void test_one_chunk_reduction_tails()
{
    // Exercise the one-vector-chunk fast path across load, tile, and residue
    // boundaries. Different row lengths also vary CSR row-start alignment.
    const int lengths[] = {0, 1, 3, 4, 5, 31, 32, 33, 63, 64, 65, 124, 125, 126,
        127, 128, 129, 255, 256, 511};
    const int rows = 7, shared = 512;
    for (int nnz : lengths) {
        float *AD = (float *) calloc(rows * shared, sizeof AD[0]);
        for (int row = 0; row < rows; row++) {
            int count = nnz > row % 4 ? nnz - row % 4 : 0;
            for (int k = 0; k < count; k++)
                AD[row * shared + k] = (float) (1 + (row + k) % 7);
        }
        // Public instantiation contracts require cols divisible by the tile.
        for (int cols : {128, 256}) {
            char label[128];
            snprintf(
                label, sizeof label, "one_chunk(nnz=%d, cols=%d)", nnz, cols);
            run_spmm(
                label, AD, rows, shared, cols, Klas_SPMM_g_spmm_f32_128x128x32);
        }
        free(AD);
    }
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--no-check") == 0) {
        do_check = false;
        argc--;
        argv++;
    }

    if (argc > 1) {
        fprintf(stderr, "Usage: %s [--no-check]\n", progname);
        return 1;
    }

    /* Square matrices, various sizes and densities.
       cols must be a multiple of 128 (blockItemsX). */
    int sizes[] = {
        128,
        256,
        512,
        1024,
    };
    int densities[] = {1, 10, 50, 100};

#define ARRLEN(s) (sizeof(s) / sizeof(s[0]))

    for (int si = 0; si < ARRLEN(sizes); si++)
        for (int di = 0; di < ARRLEN(densities); di++)
            test_random(sizes[si], sizes[si], sizes[si], densities[di]);

    /* Non-square matrices */
    test_random(256, 512, 128, 10);
    test_random(512, 128, 256, 10);
    test_random(128, 1024, 256, 5);
    test_random(1024, 256, 128, 20);
    test_random(2048, 256, 128, 10);
    test_random(128, 256, 1024, 10);

    // /* Edge cases */
    test_empty(128, 128, 128);
    test_empty(256, 512, 128);
    test_identity(128, 128);
    test_identity(256, 256);
    test_identity(512, 128);
    test_single_per_row(128, 256, 128);
    test_single_per_row(256, 512, 256);
    test_single_per_row(1024, 1024, 128);
    test_one_chunk_reduction_tails();

    printf("%d tests, %s\n", g_tests, g_ok ? "OK" : "FAILED");
    return g_ok ? 0 : 1;
}
