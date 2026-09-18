/* Common SPMM matrix generation utilities.
 * Shared between Test_Klas_SPMM.cu and Tune_Klas_SPMM.cu */

#include "Klas_SPMM.h"
#include "test-common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <numeric>
#include <algorithm>

typedef Kuiper_Sparse_Matrix_smatrix__uint32 smatrix_u32_t;
typedef Kuiper_Sparse_Matrix_smatrix__float32 smatrix_f32_t;

/* The CSR index arrays (col_ind, row_off) and the row permutation are `sz` in
 * the F* source.  How that is spelled in C depends on --custard_sizet_width
 * (see custard.mk), so take it from the extracted struct rather than restating
 * it here. */
typedef __typeof__(((smatrix_u32_t *) 0)->col_ind[0]) spmm_idx_t;

static uint32_t *mk_dense_matrix_u32(int rows, int cols, int density_pct)
{
    uint32_t *M = (uint32_t *) calloc(rows * cols, sizeof M[0]);
    for (int i = 0; i < rows * cols; i++) {
        if (rand() % 100 < density_pct)
            M[i] = 1 + rand() % 99;
    }
    return M;
}

static float *mk_dense_matrix_f32(int rows, int cols, int density_pct)
{
    float *M = (float *) calloc(rows * cols, sizeof M[0]);
    for (int i = 0; i < rows * cols; i++) {
        if (rand() % 100 < density_pct)
            M[i] = (float) (1 + rand() % 99);
    }
    return M;
}

/* Identity-like: A[i][i] = 1 for min(rows,cols) diagonal entries. */
static uint32_t *mk_identity_matrix_u32(int rows, int cols)
{
    uint32_t *M = (uint32_t *) calloc(rows * cols, sizeof M[0]);
    int diag = rows < cols ? rows : cols;
    for (int i = 0; i < diag; i++)
        M[i * cols + i] = 1;
    return M;
}

static float *mk_identity_matrix_f32(int rows, int cols)
{
    float *M = (float *) calloc(rows * cols, sizeof M[0]);
    int diag = rows < cols ? rows : cols;
    for (int i = 0; i < diag; i++)
        M[i * cols + i] = 1;
    return M;
}

/* One nonzero per row, at a random column. */
static uint32_t *mk_single_per_row_u32(int rows, int cols)
{
    uint32_t *M = (uint32_t *) calloc(rows * cols, sizeof M[0]);
    for (int i = 0; i < rows; i++)
        M[i * cols + rand() % cols] = 1 + rand() % 99;
    return M;
}

static float *mk_single_per_row_f32(int rows, int cols)
{
    float *M = (float *) calloc(rows * cols, sizeof M[0]);
    for (int i = 0; i < rows; i++)
        M[i * cols + rand() % cols] = 1 + rand() % 99;
    return M;
}

static smatrix_u32_t sparsify_u32(uint32_t *M, int rows, int cols)
{
    uint32_t nnz = 0;
    for (int i = 0; i < rows * cols; i++)
        if (M[i] != 0)
            nnz++;

    uint32_t *elems =
        (uint32_t *) malloc((nnz > 0 ? nnz : 1) * sizeof elems[0]);
    spmm_idx_t *col_ind =
        (spmm_idx_t *) malloc((nnz > 0 ? nnz : 1) * sizeof col_ind[0]);
    spmm_idx_t *row_off = (spmm_idx_t *) malloc((rows + 1) * sizeof row_off[0]);

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

    smatrix_u32_t s;
    s.nnz = nnz;
    s.elems = elems;
    s.col_ind = col_ind;
    s.row_off = row_off;
    return s;
}

static smatrix_f32_t sparsify_f32(float *M, int rows, int cols)
{
    uint32_t nnz = 0;
    for (int i = 0; i < rows * cols; i++)
        if (M[i] != 0.0f)
            nnz++;

    float *elems = (float *) malloc((nnz > 0 ? nnz : 1) * sizeof elems[0]);
    spmm_idx_t *col_ind =
        (spmm_idx_t *) malloc((nnz > 0 ? nnz : 1) * sizeof col_ind[0]);
    spmm_idx_t *row_off = (spmm_idx_t *) malloc((rows + 1) * sizeof row_off[0]);

    uint32_t idx = 0;
    row_off[0] = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (M[i * cols + j] != 0.0f) {
                elems[idx] = M[i * cols + j];
                col_ind[idx] = j;
                idx++;
            }
        }
        row_off[i + 1] = idx;
    }
    assert(idx == nnz);

    smatrix_f32_t s;
    s.nnz = nnz;
    s.elems = elems;
    s.col_ind = col_ind;
    s.row_off = row_off;
    return s;
}

template <typename smatrix_t>
static spmm_idx_t *mk_row_indices(int rows, smatrix_t A)
{
    spmm_idx_t *row_indices =
        (spmm_idx_t *) malloc(rows * sizeof row_indices[0]);

    std::vector<spmm_idx_t> swizzle_staging(rows);
    std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

    std::sort(swizzle_staging.begin(), swizzle_staging.end(),
        [&A](int idx_a, int idx_b) {
            spmm_idx_t length_a = A.row_off[idx_a + 1] - A.row_off[idx_a];
            spmm_idx_t length_b = A.row_off[idx_b + 1] - A.row_off[idx_b];
            return length_a > length_b;
        });

    memcpy(row_indices, swizzle_staging.data(), sizeof(row_indices[0]) * rows);

    return row_indices;
}

/* Upload a u32 sparse matrix + dense B to the GPU. Caller must free device
 * pointers. */
static void upload_spmm_u32(int rows, int shared, int cols, smatrix_u32_t A,
    spmm_idx_t *row_indices, uint32_t *B, smatrix_u32_t *dA_out,
    spmm_idx_t **drow_indices_out, uint32_t **dB_out, uint32_t **dC_out)
{
    smatrix_u32_t dA;
    dA.nnz = A.nnz;
    dA.elems =
        (uint32_t *) kpr_wait_alloc(sizeof dA.elems[0], A.nnz > 0 ? A.nnz : 1);
    dA.col_ind = (spmm_idx_t *) kpr_wait_alloc(
        sizeof dA.col_ind[0], A.nnz > 0 ? A.nnz : 1);
    dA.row_off = (spmm_idx_t *) kpr_wait_alloc(sizeof dA.row_off[0], rows + 1);

    if (A.nnz > 0) {
        MUST(cudaMemcpy(dA.elems, A.elems, sizeof A.elems[0] * A.nnz,
            cudaMemcpyHostToDevice));
        MUST(cudaMemcpy(dA.col_ind, A.col_ind, sizeof A.col_ind[0] * A.nnz,
            cudaMemcpyHostToDevice));
    }
    MUST(cudaMemcpy(dA.row_off, A.row_off, sizeof A.row_off[0] * (rows + 1),
        cudaMemcpyHostToDevice));

    uint32_t *dB = (uint32_t *) kpr_wait_alloc(sizeof dB[0], shared * cols);
    MUST(
        cudaMemcpy(dB, B, sizeof B[0] * shared * cols, cudaMemcpyHostToDevice));
    spmm_idx_t *drow_indices =
        (spmm_idx_t *) kpr_wait_alloc(sizeof row_indices[0], rows);
    MUST(cudaMemcpy(drow_indices, row_indices, sizeof row_indices[0] * rows,
        cudaMemcpyHostToDevice));
    uint32_t *dC = (uint32_t *) kpr_wait_alloc(sizeof dC[0], rows * cols);
    MUST(cudaMemset(dC, 0, sizeof dC[0] * rows * cols));

    *dA_out = dA;
    *drow_indices_out = drow_indices;
    *dB_out = dB;
    *dC_out = dC;
}

/* Upload a f32 sparse matrix + dense B to the GPU. Caller must free device
 * pointers. */
static void upload_spmm_f32(int rows, int shared, int cols, smatrix_f32_t A,
    spmm_idx_t *row_indices, float *B, smatrix_f32_t *dA_out,
    spmm_idx_t **drow_indices_out, float **dB_out, float **dC_out)
{
    smatrix_f32_t dA;
    dA.nnz = A.nnz;
    dA.elems =
        (float *) kpr_wait_alloc(sizeof dA.elems[0], A.nnz > 0 ? A.nnz : 1);
    dA.col_ind = (spmm_idx_t *) kpr_wait_alloc(
        sizeof dA.col_ind[0], A.nnz > 0 ? A.nnz : 1);
    dA.row_off = (spmm_idx_t *) kpr_wait_alloc(sizeof dA.row_off[0], rows + 1);

    if (A.nnz > 0) {
        MUST(cudaMemcpy(dA.elems, A.elems, sizeof A.elems[0] * A.nnz,
            cudaMemcpyHostToDevice));
        MUST(cudaMemcpy(dA.col_ind, A.col_ind, sizeof A.col_ind[0] * A.nnz,
            cudaMemcpyHostToDevice));
    }
    MUST(cudaMemcpy(dA.row_off, A.row_off, sizeof A.row_off[0] * (rows + 1),
        cudaMemcpyHostToDevice));

    float *dB = (float *) kpr_wait_alloc(sizeof dB[0], shared * cols);
    MUST(
        cudaMemcpy(dB, B, sizeof B[0] * shared * cols, cudaMemcpyHostToDevice));
    spmm_idx_t *drow_indices =
        (spmm_idx_t *) kpr_wait_alloc(sizeof row_indices[0], rows);
    MUST(cudaMemcpy(drow_indices, row_indices, sizeof row_indices[0] * rows,
        cudaMemcpyHostToDevice));
    float *dC = (float *) kpr_wait_alloc(sizeof dC[0], rows * cols);
    MUST(cudaMemset(dC, 0, sizeof dC[0] * rows * cols));

    *dA_out = dA;
    *drow_indices_out = drow_indices;
    *dB_out = dB;
    *dC_out = dC;
}

static void free_spmm_device_u32(
    smatrix_u32_t dA, spmm_idx_t *drow_indices, uint32_t *dB, uint32_t *dC)
{
    cudaFree(dA.elems);
    cudaFree(dA.col_ind);
    cudaFree(dA.row_off);
    cudaFree(drow_indices);
    cudaFree(dB);
    cudaFree(dC);
}

static void free_spmm_device_f32(
    smatrix_f32_t dA, spmm_idx_t *drow_indices, float *dB, float *dC)
{
    cudaFree(dA.elems);
    cudaFree(dA.col_ind);
    cudaFree(dA.row_off);
    cudaFree(drow_indices);
    cudaFree(dB);
    cudaFree(dC);
}
