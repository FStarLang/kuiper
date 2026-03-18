#include <unistd.h>
#include "test-common.h"

const char *progname = NULL;

#ifndef cet
#define cet et
#endif

double et2double(et v)
{
#if et_is_half == 1
    return __half2float(v);
#else
    return v;
#endif
}

// for tensor core operations assume the accumulator's type for inputs as well
cet *cpu_mul(size_t rows, size_t shared, size_t cols, et *m1, et *m2)
{
    // Accumulate in double to avoid precision loss from repeated
    // rounding to cet (especially when cet is half).
    // Note: tensor cores accumulate internally in FP32.
    double *acc = (double *) calloc(rows * cols, sizeof(double));
    int i, j, k;
    for (i = 0; i < rows; i++) {
        for (k = 0; k < shared; k++) {
            for (j = 0; j < cols; j++) {
                acc[i * cols + j] +=
                    et2double(m1[i * shared + k]) * et2double(m2[k * cols + j]);
            }
        }
    }
    cet *m3 = (cet *) malloc(rows * cols * sizeof(cet));
    for (size_t n = 0; n < rows * cols; n++) {
        m3[n] = acc[n];
    }
    free(acc);
    return m3;
}

cet *cpu_sgemm(size_t rows, size_t shared, size_t cols, et *m1_cpu, et *m2_cpu,
               cet *m3_cpu, cet alpha, cet beta)
{
    cet *mm = cpu_mul(rows, shared, cols, m1_cpu, m2_cpu);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mm[i * cols + j] =
                alpha * mm[i * cols + j] + beta * m3_cpu[i * cols + j];
            // printf("GG gemm @ (%d,%d) = %g\n", i, j, __half2float(mm[i * cols + j]));
        }
    }
    return mm;
}

/* m1 is GPU-side result, m2 is CPU-computed result */
void cmp(size_t rows, size_t cols, cet *m1_gpu, cet *m2)
{
    fprintf(stderr, "Checking result...\n");
    int nerrs = 0;
    int i, j;
    double max_rel = 0.0;
    double sum_rel = 0.0;
    size_t nrel = 0;
    cet *m1 = (cet *) calloc(sizeof(cet), rows * cols);
    MUST(cudaMemcpy
         (m1, m1_gpu, rows * cols * sizeof(cet), cudaMemcpyDeviceToHost));
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            double vm1 = et2double(m1[i * cols + j]);
            double vm2 = et2double(m2[i * cols + j]);

            double diff = vm1 - vm2;
            if (vm2 != 0.0) {
                double rel = fabs(diff / vm2);
                if (rel > max_rel)
                    max_rel = rel;
                sum_rel += rel;
                nrel++;
            }
            bool ok = vm1 == vm2;
#ifndef EXACT
            /* Check, also, if the relative error is less than TOLERANCE */
            ok = ok || (fabs(diff / vm2) <= TOLERANCE);
#endif
            if (!ok) {
                fprintf(stderr, "diff=%g, vm1=%g, vm2=%g\n", diff, vm1, vm2);
                fprintf(stderr, "Error at pos (%d, %d): %g != %g\n", i, j, vm1,
                        vm2);
                fprintf(stderr,
                        "::error::%s -- Error at pos (%d, %d): %g (gpu) != %g (cpu)\n",
                        progname, i, j, vm1, vm2);
                nerrs++;
                if (nerrs >= 10) {
                    fprintf(stderr, ">> max relative error: %.4f%%, avg relative error: %.4f%% (%zu cells)\n",
                            max_rel * 100.0, nrel > 0 ? sum_rel / nrel * 100.0 : 0.0, nrel);
                    free(m1);
                    exit(1);
                }
            }
        }
    }
    fprintf(stderr, ">> max relative error: %.4f%%, avg relative error: %.4f%% (%zu cells)\n",
            max_rel * 100.0, nrel > 0 ? sum_rel / nrel * 100.0 : 0.0, nrel);
    free(m1);
}
