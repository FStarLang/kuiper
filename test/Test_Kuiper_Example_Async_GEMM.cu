#include "Kuiper_Example_Async_GEMM.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define et        float
#define et_lbl    f32
#define TOLERANCE 0.001f

#include "matmul_common.c.inc"

#define DIM 1024

int main()
{
    int i, j;

    float *a_cpu = (float *)malloc(DIM * DIM * sizeof(float));
    float *b_cpu = (float *)malloc(DIM * DIM * sizeof(float));
    float *c_cpu = (float *)malloc(DIM * DIM * sizeof(float));
    float *d_cpu = (float *)malloc(DIM * DIM * sizeof(float));

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            a_cpu[i * DIM + j] = (float)((2 * i + j) % 7);
            b_cpu[i * DIM + j] = (float)((i + j) % 5);
            c_cpu[i * DIM + j] = (float)((i + 2 * j) % 7);
            d_cpu[i * DIM + j] = (float)((2 * i + j) % 5);
        }
    }

    /* CPU reference: (A*B) * (C*D) */
    float *ab = cpu_mul(DIM, DIM, DIM, a_cpu, b_cpu);
    float *cd = cpu_mul(DIM, DIM, DIM, c_cpu, d_cpu);
    float *expected = cpu_mul(DIM, DIM, DIM, ab, cd);

    /* GPU */
    float *a = (float *)kpr_wait_alloc(sizeof(float), DIM * DIM);
    float *b = (float *)kpr_wait_alloc(sizeof(float), DIM * DIM);
    float *c = (float *)kpr_wait_alloc(sizeof(float), DIM * DIM);
    float *d = (float *)kpr_wait_alloc(sizeof(float), DIM * DIM);
    float *r = (float *)kpr_wait_alloc(sizeof(float), DIM * DIM);

    MUST(cudaMemcpy(a, a_cpu, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(b, b_cpu, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(c, c_cpu, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(d, d_cpu, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));

    Kuiper_Example_Async_GEMM_main(a, b, c, d, r);

    cmp(DIM, DIM, r, expected);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(r);
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);
    free(d_cpu);
    free(ab);
    free(cd);
    free(expected);

    printf("OK\n");

    return 0;
}
