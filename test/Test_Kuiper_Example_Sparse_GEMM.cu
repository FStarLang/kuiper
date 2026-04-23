#include "Kuiper_Example_Sparse_GEMM.h"
#include "test-common.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

const char *progname = __FILE__;

#define N 128

typedef Kuiper_Sparse_Matrix_smatrix__uint32_t smatrix_t;

uint32_t *mk_dense_matrix()
{
    uint32_t *M = (uint32_t *) malloc(N * N * sizeof M[0]);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i * N + j] = rand() % 100 < 90 ? 0 : rand() % 100;
        }
    }
    return M;
}

smatrix_t sparsify_matrix(uint32_t *M)
{
    uint32_t nnz = 0;
    for (int i = 0; i < N * N; i++) {
        if (M[i] != 0) {
            nnz++;
        }
    }

    uint32_t *elems = (uint32_t *) malloc(nnz * sizeof elems[0]);
    uint32_t *col_ind = (uint32_t *) malloc(nnz * sizeof col_ind[0]);
    uint32_t *row_off = (uint32_t *) malloc((N + 1) * sizeof row_off[0]);

    uint32_t idx = 0;
    row_off[0] = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (M[i * N + j] != 0) {
                elems[idx] = M[i * N + j];
                col_ind[idx] = j;
                idx++;
            }
        }
        row_off[i + 1] = idx;
    }
    assert(idx == nnz);

    Kuiper_Sparse_Matrix_smatrix__uint32_t smat;
    smat.nnz = nnz;
    smat.elems = elems;
    smat.col_ind = col_ind;
    smat.row_off = row_off;

    return smat;
}

void matmul(uint32_t *A, uint32_t *B, uint32_t *C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            uint32_t sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void print_matrix(const char *name, uint32_t *M)
{
    printf("Matrix %s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5u ", M[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // srand(time(NULL) + getpid());
    uint32_t *AD = mk_dense_matrix();
    smatrix_t A = sparsify_matrix(AD);
    uint32_t *B = mk_dense_matrix();
    uint32_t *CD = (uint32_t *) calloc(N * N, sizeof CD[0]);

    smatrix_t dA;
    dA.nnz = A.nnz;
    dA.elems = (uint32_t *) kpr_wait_alloc(sizeof dA.elems[0], A.nnz);
    dA.col_ind = (uint32_t *) kpr_wait_alloc(sizeof dA.col_ind[0], A.nnz);
    dA.row_off = (uint32_t *) kpr_wait_alloc(sizeof dA.row_off[0], N + 1);

    cudaMemcpy(dA.elems, A.elems, sizeof A.elems[0] * A.nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dA.col_ind, A.col_ind, sizeof A.col_ind[0] * A.nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dA.row_off, A.row_off, sizeof A.row_off[0] * (N + 1), cudaMemcpyHostToDevice);

    uint32_t *dB = (uint32_t *) kpr_wait_alloc(sizeof dB[0], N * N);
    cudaMemcpy(dB, B, sizeof B[0] * N * N, cudaMemcpyHostToDevice);
    uint32_t *dC = (uint32_t *) kpr_wait_alloc(sizeof dC[0], N * N);

    Kuiper_Example_Sparse_GEMM__gemm_u32_rr(N, N, N, dA, dB, dC);
    matmul(AD, B, CD);

    uint32_t *C = (uint32_t *) calloc(N * N, sizeof C[0]);
    cudaMemcpy(C, dC, sizeof C[0] * N * N, cudaMemcpyDeviceToHost);

    cudaFree(dA.elems);
    cudaFree(dA.col_ind);
    cudaFree(dA.row_off);
    cudaFree(dB);
    cudaFree(dC);

    // print_matrix("AD", AD);
    // print_matrix("B", B);
    // print_matrix("C (sparse gemm)", C);
    // print_matrix("CD (dense gemm)", CD);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] != CD[i * N + j]) {
                printf("Mismatch at (%d,%d): %u != %u\n", i, j, C[i * N + j], CD[i * N + j]);
                return 1;
            }
        }
    }

    printf("Done\n");

    free(B);
    free(C);
    free(A.elems);
    free(A.col_ind);
    free(A.row_off);
    free(AD);

    return 0;
}
