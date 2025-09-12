#include <unistd.h>
#include "test-common.h"

const char *progname = NULL;

et *cpu_mul(size_t rows, size_t shared, size_t cols, et *m1, et *m2)
{
    et *m3 = (et *) calloc(rows * cols, sizeof(et));
    int i, j, k;
    for (i = 0; i < rows; i++) {
        for (k = 0; k < shared; k++) {
            for (j = 0; j < cols; j++) {
                m3[i * cols + j] += m1[i * shared + k] * m2[k * cols + j];
            }
        }
    }
    return m3;
}

et *cpu_sgemm(size_t rows, size_t shared, size_t cols, et *m1_cpu, et *m2_cpu,
              et *m3_cpu, et alpha, et beta)
{
    et *mm = cpu_mul(rows, shared, cols, m1_cpu, m2_cpu);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mm[i * cols + j] =
                alpha * mm[i * cols + j] + beta * m3_cpu[i * cols + j];
        }
    }
    return mm;
}

/* m1 is GPU-side result, m2 is CPU-computed result */
void cmp(size_t rows, size_t cols, et *m1_gpu, et *m2)
{
    fprintf(stderr, "Checking result...\n");
    int nerrs = 0;
    int i, j;
    et *m1 = (et *) calloc(sizeof(et), rows * cols);
    MUST(cudaMemcpy
         (m1, m1_gpu, rows * cols * sizeof(et), cudaMemcpyDeviceToHost));
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            bool ok;
#ifdef EXACT
            ok = m1[i * cols + j] == m2[i * cols + j];
#else
            /* Check if the relative error is less than TOLERANCE */
            et diff = m1[i * cols + j] - m2[i * cols + j];
            ok = fabs((double)diff / (double)m2[i * cols + j]) <= TOLERANCE;
#endif
            if (!ok) {
#if et == half
                float vm1 = __half2float(m1[i * cols + j]);
                float vm2 = __half2float(m2[i * cols + j]);
#else
                et vm1 = m1[i * cols + j];
                et vm2 = m2[i * cols + j];
#endif
                fprintf(stderr, "Error at pos (%d, %d): %" PRIet " != %"
                        PRIet "\n", i, j, vm1, vm2);
                fprintf(stderr,
                        "::error::%s -- Error at pos (%d, %d): %" PRIet " != %"
                        PRIet "\n", progname, i, j, vm1, vm2);
                nerrs++;
                if (nerrs >= 10) {
                    free(m1);
                    exit(1);
                }
            }
        }
    }
    free(m1);
}
