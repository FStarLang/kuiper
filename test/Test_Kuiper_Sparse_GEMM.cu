#include "Kuiper_Sparse_GEMM.h"
#include "timing.h"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define N 128

typedef Kuiper_Sparse_smatrix__uint32_t smatrix_t;

uint32_t *mk_dense_matrix()
{
    uint32_t* M = (uint32_t*)malloc (N * N * sizeof M[0]);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i * N + j] = 
              rand () % 100 < 90 ? 0 : rand() % 100;
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

    uint32_t* elems = (uint32_t*)malloc (nnz * sizeof elems[0]);
    uint32_t* col_ind = (uint32_t*)malloc (nnz * sizeof col_ind[0]);
    uint32_t* row_off = (uint32_t*)malloc ((N + 1) * sizeof row_off[0]);

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
    assert (idx == nnz);

    Kuiper_Sparse_smatrix__uint32_t smat;
    smat.nnz1 = nnz;
    smat.elems1 = elems;
    smat.col_ind = col_ind;
    smat.row_off = row_off;

    return smat;
}

void matmul(uint32_t* A, uint32_t* B, uint32_t* C)
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

void print_matrix(const char *name, uint32_t* M)
{
    printf ("Matrix %s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf ("%5u ", M[i * N + j]);
        }
        printf ("\n");
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL) + getpid());
    uint32_t *AD = mk_dense_matrix();
    smatrix_t A = sparsify_matrix(AD);
    uint32_t* B = mk_dense_matrix();
    uint32_t* C = (uint32_t*)calloc (N * N, sizeof C[0]);
    uint32_t* CD = (uint32_t*)calloc (N * N, sizeof C[0]);

    Kuiper_Sparse_GEMM__gemm_u32_rr(N, N, N, A, B, C);
    matmul(AD, B, CD);

    // print_matrix("AD", AD);
    // print_matrix("B", B);
    // print_matrix("C (sparse gemm)", C);
    // print_matrix("CD (dense gemm)", CD);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] != CD[i * N + j]) {
                printf ("Mismatch at (%d,%d): %u != %u\n",
                        i, j, C[i * N + j], CD[i * N + j]);
                return 1;
            }
        }
    }

    printf("Done\n");

    free(B);
    free(C);
    free(A.elems1);
    free(A.col_ind);
    free(A.row_off);
    free(AD);

    return 0;
}
