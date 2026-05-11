#include "spmm_common.c.inc"

const char *progname = __FILE__;

typedef Kuiper_Sparse_Matrix_smatrix__uint32_t smatrix_t;

static int g_ok = 1;
static int g_tests = 0;
static bool do_check = 1;

static void cpu_matmul(uint32_t *A, uint32_t *B, uint32_t *C, int rows, int shared, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            uint32_t sum = 0;
            for (int k = 0; k < shared; k++)
                sum += A[i * shared + k] * B[k * cols + j];
            C[i * cols + j] = sum;
        }
}

static void run_spmm(const char *label, uint32_t *AD, int rows, int shared, int cols)
{
    smatrix_t A = sparsify_u32(AD, rows, shared);
    uint32_t *row_indices = mk_row_indices(rows, A);
    uint32_t *B = mk_dense_matrix_u32(shared, cols, 50);
    uint32_t *CD = (uint32_t *) calloc(rows * cols, sizeof CD[0]);
    cpu_matmul(AD, B, CD, rows, shared, cols);

    smatrix_t dA;
    uint32_t *drow_indices, *dB, *dC;
    upload_spmm_u32(rows, shared, cols, A, row_indices, B, &dA, &drow_indices, &dB, &dC);

    float t;
    TIME_void(Klas_SPMM_spmm_u32(rows, shared, cols, dA, drow_indices, dB, dC), &t);
    fprintf(stderr, ">>> RES (rows=%d, shared=%d, cols=%d, sparsity=%.2f%%) \t GFLOPS: %.3f\n",
            rows, shared, cols,
            (1.0 - (double)A.nnz / (rows * shared)) * 100.0, (A.nnz * shared * 2.0) / t / 1e9);

    uint32_t *C = (uint32_t *) calloc(rows * cols, sizeof C[0]);
    MUST(cudaMemcpy(C, dC, sizeof C[0] * rows * cols, cudaMemcpyDeviceToHost));

    free_spmm_device_u32(dA, drow_indices, dB, dC);

    g_tests++;
    int mismatches = 0;
    if (!do_check) {
        for (int i = 0; i < rows * cols; i++) {
            if (C[i] != CD[i]) {
                if (mismatches == 0)
                    fprintf(stderr, "FAIL %s: first mismatch at (%d,%d): "
                            "got %u, ref %u\n", label, i / cols, i % cols, C[i], CD[i]);
                mismatches++;
            }
        }
        if (mismatches > 0) {
            fprintf(stderr, "FAIL %s: %d mismatches out of %d\n", label, mismatches, rows * cols);
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
    snprintf(label, sizeof label, "random(%dx%dx%d, %d%%)", rows, shared, cols, density_pct);
    uint32_t *AD = mk_dense_matrix_u32(rows, shared, density_pct);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

static void test_identity(int n, int cols)
{
    char label[128];
    snprintf(label, sizeof label, "identity(%dx%d, cols=%d)", n, n, cols);
    uint32_t *AD = mk_identity_matrix(n, n);
    run_spmm(label, AD, n, n, cols);
    free(AD);
}

static void test_empty(int rows, int shared, int cols)
{
    char label[128];
    snprintf(label, sizeof label, "empty(%dx%dx%d)", rows, shared, cols);
    uint32_t *AD = (uint32_t *) calloc(rows * shared, sizeof AD[0]);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

static void test_single_per_row(int rows, int shared, int cols)
{
    char label[128];
    snprintf(label, sizeof label, "single_per_row(%dx%dx%d)", rows, shared, cols);
    uint32_t *AD = mk_single_per_row(rows, shared);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
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
    int sizes[] = { 128, 256, 512, 1024 };
    int densities[] = { 1, 10, 50, 100 };

#define ARRLEN(s) (sizeof(s)/sizeof(s[0]))

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

    /* Edge cases */
    test_empty(128, 128, 128);
    test_empty(256, 512, 128);
    test_identity(128, 128);
    test_identity(256, 256);
    test_identity(512, 128);
    test_single_per_row(128, 256, 128);
    test_single_per_row(256, 512, 256);
    test_single_per_row(1024, 1024, 128);

    printf("%d tests, %s\n", g_tests, g_ok ? "OK" : "FAILED");
    return g_ok ? 0 : 1;
}
