#include "Klas_GEMM_Naive1.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "test-common.h"

const char *progname = "Test_Klas_GEMM_Naive1__F32_Batched";

#define TOLERANCE 0.001f

static float *cpu_batched_mul(uint32_t batch, uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              float *m1, float *m2)
{
    float *out = (float *)calloc(batch * m * n, sizeof(float));
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t l = 0; l < k; l++) {
                for (uint32_t j = 0; j < n; j++) {
                    out[b * (m * n) + i * n + j] +=
                        m1[b * (m * k) + i * k + l] *
                        m2[b * (k * n) + l * n + j];
                }
            }
        }
    }
    return out;
}

int main(int argc, char **argv)
{
    uint32_t batch = 4;
    uint32_t m = 64;
    uint32_t k = 32;
    uint32_t n = 64;

    printf("Batch = %u\n", batch);
    printf("M = %u\n", m);
    printf("N = %u\n", n);
    printf("K = %u\n", k);

    size_t a_elems = batch * m * k;
    size_t b_elems = batch * k * n;
    size_t c_elems = batch * m * n;

    float *a_cpu = (float *)malloc(a_elems * sizeof(float));
    float *b_cpu = (float *)malloc(b_elems * sizeof(float));

    for (uint32_t bi = 0; bi < batch; bi++) {
        for (uint32_t i = 0; i < m; i++)
            for (uint32_t j = 0; j < k; j++)
                a_cpu[bi * (m * k) + i * k + j] = (float)(2 * i + j + bi);
        for (uint32_t i = 0; i < k; i++)
            for (uint32_t j = 0; j < n; j++)
                b_cpu[bi * (k * n) + i * n + j] = (float)(i + j + bi);
    }

    float *a_gpu = (float *)kpr_wait_alloc(sizeof(float), a_elems);
    float *b_gpu = (float *)kpr_wait_alloc(sizeof(float), b_elems);
    float *c_gpu = (float *)kpr_wait_alloc(sizeof(float), c_elems);

    cudaMemcpy(a_gpu, a_cpu, a_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, b_elems * sizeof(float), cudaMemcpyHostToDevice);

    float *expected = cpu_batched_mul(batch, m, n, k, a_cpu, b_cpu);

    Klas_GEMM_Naive1_batched_matmul_f32(batch, m, n, k, a_gpu, b_gpu, c_gpu);

    float *c_host = (float *)malloc(c_elems * sizeof(float));
    MUST(cudaMemcpy(c_host, c_gpu, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    int nerrs = 0;
    for (uint32_t bi = 0; bi < batch; bi++) {
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < n; j++) {
                size_t idx = bi * (m * n) + i * n + j;
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

    float *c_initial = (float *)malloc(c_elems * sizeof(float));
    for (size_t i = 0; i < c_elems; i++)
        c_initial[i] = (float)(i % 7);

    float *gemm_gpu = (float *)kpr_wait_alloc(sizeof(float), c_elems);
    cudaMemcpy(gemm_gpu, c_initial, c_elems * sizeof(float), cudaMemcpyHostToDevice);
    Klas_GEMM_Naive1_batched_gemm_f32(2.0f, 0.5f, batch, m, n, k,
                                       a_gpu, b_gpu, gemm_gpu);
    MUST(cudaMemcpy(c_host, gemm_gpu, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < c_elems; i++) {
        float exp = 0.5f * c_initial[i] + 2.0f * expected[i];
        float got = c_host[i];
        bool ok = (got == exp) || (exp != 0.0f && fabsf((got - exp) / exp) <= TOLERANCE);
        if (!ok) {
            fprintf(stderr, "GEMM error at index=%zu: %g (gpu) != %g (cpu)\n",
                    i, got, exp);
            nerrs++;
            if (nerrs >= 10)
                return 1;
        }
    }

    cudaFree(gemm_gpu);
    cudaFree(c_gpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    free(a_cpu);
    free(b_cpu);
    free(expected);
    free(c_initial);
    free(c_host);

    printf("OK\n");
    return 0;
}
