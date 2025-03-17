#include <unistd.h>

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

/* m1 is GPU-side result, m2 is CPU-computed result */
void cmp(size_t rows, size_t cols, et *m1_gpu, et *m2)
{
    int i, j;
    et *m1 = (et *) calloc(sizeof(et), rows * cols);
    cudaMemcpy(m1, m1_gpu, rows * cols * sizeof(et), cudaMemcpyDeviceToHost);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            bool ok;
#ifdef EXACT
            ok = m1[i * cols + j] == m2[i * cols + j];
#else
            /* Check if the relative error is less than TOLERANCE */
            et diff = m1[i * cols + j] - m2[i * cols + j];
            ok = (diff / (double)m2[i * cols + j]) <= TOLERANCE;
#endif
            if (!ok) {
                fprintf(stderr, "Error at pos (%d, %d): %" PRIet " != %"
                        PRIet "\n", i, j, m1[i * cols + j], m2[i * cols + j]);
                fprintf(stderr,
                        "::error::%s -- Error at pos (%d, %d): %" PRIet " != %"
                        PRIet "\n", progname, i, j, m1[i * cols + j],
                        m2[i * cols + j]);
                free(m1);
                exit(1);
            }
        }
    }
    free(m1);
}

/*
 * When running several tests in parallel, we may exhaust the memory
 * of the GPU. This silly wrapper will just wait and keep retrying if
 * the allocation fails.
 */
static void *kpr_wait_alloc(size_t sz, size_t len)
{
    void *ret;
    cudaError rc;
    int tries = 10;

    while ((rc = cudaMalloc(&ret, sz * len)) == cudaErrorMemoryAllocation &&
           tries-- > 0) {
        fprintf(stderr, "%s: Waiting for GPU memory...\n", progname);
        sleep(5);
    }
    MUST(rc);
    return ret;
}
