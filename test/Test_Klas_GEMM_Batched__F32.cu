#include "Klas_GEMM_Batched.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "test-common.h"

const char *progname = "Test_Klas_GEMM_Batched__F32";

#define TOLERANCE 0.001f

static float *cpu_batched_mul(uint32_t batch, uint32_t rows,
                              uint32_t shared, uint32_t cols, float *m1, float *m2)
{
    float *out = (float *)calloc(batch * rows * cols, sizeof(float));
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t i = 0; i < rows; i++) {
            for (uint32_t k = 0; k < shared; k++) {
                for (uint32_t j = 0; j < cols; j++) {
                    out[b * (rows * cols) + i * cols + j] +=
                        m1[b * (rows * shared) + i * shared + k] *
                        m2[b * (shared * cols) + k * cols + j];
                }
            }
        }
    }
    return out;
}

int main(int argc, char **argv)
{
    uint32_t batch = 4;
    uint32_t rows = 64;
    uint32_t shared = 32;
    uint32_t cols = 64;

    printf("Batch = %u\n", batch);
    printf("Rows = %u\n", rows);
    printf("Shared = %u\n", shared);
    printf("Columns = %u\n", cols);

    size_t a_elems = batch * rows * shared;
    size_t b_elems = batch * shared * cols;
    size_t c_elems = batch * rows * cols;

    float *a_cpu = (float *)malloc(a_elems * sizeof(float));
    float *b_cpu = (float *)malloc(b_elems * sizeof(float));

    for (uint32_t bi = 0; bi < batch; bi++) {
        for (uint32_t i = 0; i < rows; i++)
            for (uint32_t j = 0; j < shared; j++)
                a_cpu[bi * (rows * shared) + i * shared + j] = (float)(2 * i + j + bi);
        for (uint32_t i = 0; i < shared; i++)
            for (uint32_t j = 0; j < cols; j++)
                b_cpu[bi * (shared * cols) + i * cols + j] = (float)(i + j + bi);
    }

    float *a_gpu = (float *)kpr_wait_alloc(sizeof(float), a_elems);
    float *b_gpu = (float *)kpr_wait_alloc(sizeof(float), b_elems);

    cudaMemcpy(a_gpu, a_cpu, a_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, b_elems * sizeof(float), cudaMemcpyHostToDevice);

    float *expected = cpu_batched_mul(batch, rows, shared, cols, a_cpu, b_cpu);

    float *c_gpu = Klas_GEMM_Batched_batched_gemm_f32(batch, rows, shared, cols,
                                                      a_gpu, b_gpu);

    float *c_host = (float *)malloc(c_elems * sizeof(float));
    MUST(cudaMemcpy(c_host, c_gpu, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    int nerrs = 0;
    for (uint32_t bi = 0; bi < batch; bi++) {
        for (uint32_t i = 0; i < rows; i++) {
            for (uint32_t j = 0; j < cols; j++) {
                size_t idx = bi * (rows * cols) + i * cols + j;
                float got = c_host[idx];
                float exp = expected[idx];
                bool ok = (got == exp) || (exp != 0.0f && fabsf((got - exp) / exp) <= TOLERANCE);
                if (!ok) {
                    fprintf(stderr,
                            "Error at batch=%u pos=(%u,%u): %g (gpu) != %g (cpu)\n",
                            bi, i, j, got, exp);
                    nerrs++;
                    if (nerrs >= 10)
                        return 1;
                }
            }
        }
    }

    cudaFree(c_gpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    free(a_cpu);
    free(b_cpu);
    free(expected);
    free(c_host);

    printf("OK\n");
    return 0;
}
