#include "Kuiper_Example_Sparse_SPMM.h"
#include "test-common.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *progname = __FILE__;

typedef Kuiper_Sparse_Matrix_smatrix__uint32_t smatrix_t;

static int g_ok = 1;
static int g_tests = 0;

static uint32_t *mk_dense_matrix(int rows, int cols, int density_pct)
{
    uint32_t *M = (uint32_t *)calloc(rows * cols, sizeof M[0]);
    for (int i = 0; i < rows * cols; i++) {
        if (rand() % 100 < density_pct)
            M[i] = 1 + rand() % 99;
    }
    return M;
}

/* Identity-like: A[i][i] = 1 for min(rows,cols) diagonal entries. */
static uint32_t *mk_identity_matrix(int rows, int cols)
{
    uint32_t *M = (uint32_t *)calloc(rows * cols, sizeof M[0]);
    int diag = rows < cols ? rows : cols;
    for (int i = 0; i < diag; i++)
        M[i * cols + i] = 1;
    return M;
}

/* One nonzero per row, at a random column. */
static uint32_t *mk_single_per_row(int rows, int cols)
{
    uint32_t *M = (uint32_t *)calloc(rows * cols, sizeof M[0]);
    for (int i = 0; i < rows; i++)
        M[i * cols + rand() % cols] = 1 + rand() % 99;
    return M;
}

static smatrix_t sparsify(uint32_t *M, int rows, int cols)
{
    uint32_t nnz = 0;
    for (int i = 0; i < rows * cols; i++)
        if (M[i] != 0)
            nnz++;

    uint32_t *elems = (uint32_t *)malloc((nnz > 0 ? nnz : 1) * sizeof elems[0]);
    uint32_t *col_ind = (uint32_t *)malloc((nnz > 0 ? nnz : 1) * sizeof col_ind[0]);
    uint32_t *row_off = (uint32_t *)malloc((rows + 1) * sizeof row_off[0]);

    uint32_t idx = 0;
    row_off[0] = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (M[i * cols + j] != 0) {
                elems[idx] = M[i * cols + j];
                col_ind[idx] = j;
                idx++;
            }
        }
        row_off[i + 1] = idx;
    }
    assert(idx == nnz);

    smatrix_t s;
    s.nnz = nnz;
    s.elems = elems;
    s.col_ind = col_ind;
    s.row_off = row_off;
    return s;
}

static void cpu_matmul(uint32_t *A, uint32_t *B, uint32_t *C,
                       int rows, int shared, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            uint32_t sum = 0;
            for (int k = 0; k < shared; k++)
                sum += A[i * shared + k] * B[k * cols + j];
            C[i * cols + j] = sum;
        }
}

static void run_spmm(const char *label, uint32_t *AD,
                      int rows, int shared, int cols)
{
    smatrix_t A = sparsify(AD, rows, shared);
    uint32_t *B = mk_dense_matrix(shared, cols, 50);
    uint32_t *CD = (uint32_t *)calloc(rows * cols, sizeof CD[0]);
    cpu_matmul(AD, B, CD, rows, shared, cols);

    smatrix_t dA;
    dA.nnz = A.nnz;
    dA.elems = (uint32_t *)kpr_wait_alloc(sizeof dA.elems[0],
                                          A.nnz > 0 ? A.nnz : 1);
    dA.col_ind = (uint32_t *)kpr_wait_alloc(sizeof dA.col_ind[0],
                                            A.nnz > 0 ? A.nnz : 1);
    dA.row_off = (uint32_t *)kpr_wait_alloc(sizeof dA.row_off[0], rows + 1);

    if (A.nnz > 0) {
        MUST(cudaMemcpy(dA.elems, A.elems,
                        sizeof A.elems[0] * A.nnz, cudaMemcpyHostToDevice));
        MUST(cudaMemcpy(dA.col_ind, A.col_ind,
                        sizeof A.col_ind[0] * A.nnz, cudaMemcpyHostToDevice));
    }
    MUST(cudaMemcpy(dA.row_off, A.row_off,
                    sizeof A.row_off[0] * (rows + 1), cudaMemcpyHostToDevice));

    uint32_t *dB = (uint32_t *)kpr_wait_alloc(sizeof dB[0], shared * cols);
    MUST(cudaMemcpy(dB, B, sizeof B[0] * shared * cols, cudaMemcpyHostToDevice));
    uint32_t *dC = (uint32_t *)kpr_wait_alloc(sizeof dC[0], rows * cols);
    MUST(cudaMemset(dC, 0, sizeof dC[0] * rows * cols));

    Kuiper_Example_Sparse_SPMM_spmm_u32(rows, shared, cols, dA, dB, dC);
    MUST(cudaDeviceSynchronize());

    uint32_t *C = (uint32_t *)calloc(rows * cols, sizeof C[0]);
    MUST(cudaMemcpy(C, dC, sizeof C[0] * rows * cols, cudaMemcpyDeviceToHost));

    cudaFree(dA.elems);
    cudaFree(dA.col_ind);
    cudaFree(dA.row_off);
    cudaFree(dB);
    cudaFree(dC);

    g_tests++;
    int mismatches = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (C[i] != CD[i]) {
            if (mismatches == 0)
                fprintf(stderr, "FAIL %s: first mismatch at (%d,%d): "
                        "got %u, ref %u\n",
                        label, i / cols, i % cols, C[i], CD[i]);
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "FAIL %s: %d mismatches out of %d\n",
                label, mismatches, rows * cols);
        g_ok = 0;
    }

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
    snprintf(label, sizeof label, "random(%dx%dx%d, %d%%)",
             rows, shared, cols, density_pct);
    uint32_t *AD = mk_dense_matrix(rows, shared, density_pct);
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
    uint32_t *AD = (uint32_t *)calloc(rows * shared, sizeof AD[0]);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

static void test_single_per_row(int rows, int shared, int cols)
{
    char label[128];
    snprintf(label, sizeof label, "single_per_row(%dx%dx%d)",
             rows, shared, cols);
    uint32_t *AD = mk_single_per_row(rows, shared);
    run_spmm(label, AD, rows, shared, cols);
    free(AD);
}

int main(int argc, char **argv)
{
    /* Square matrices, various sizes and densities.
       cols must be a multiple of 128 (blockItemsX). */
    int sizes[] = { 128, 256, 512, 1024 };
    int densities[] = { 1, 10, 50, 100 };

    for (int si = 0; si < 4; si++)
        for (int di = 0; di < 4; di++)
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
