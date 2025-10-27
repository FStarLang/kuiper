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
    cet *m3 = (cet *) calloc(rows * cols, sizeof(cet));
    int i, j, k;
    for (i = 0; i < rows; i++) {
        for (k = 0; k < shared; k++) {
            for (j = 0; j < cols; j++) {
                m3[i * cols + j] +=
                    et2double(m1[i * shared + k]) * et2double(m2[k * cols + j]);
            }
        }
    }
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
    cet *m1 = (cet *) calloc(sizeof(cet), rows * cols);
    MUST(cudaMemcpy
         (m1, m1_gpu, rows * cols * sizeof(cet), cudaMemcpyDeviceToHost));
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            double vm1 = et2double(m1[i * cols + j]);
            double vm2 = et2double(m2[i * cols + j]);

            double diff = vm1 - vm2;
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
                    free(m1);
                    // printf("# of errors: %d\n", nerrs);
                    exit(1);
                }
            }
        }
    }
    // printf("# of errors: %d\n", nerrs);
    free(m1);
}
