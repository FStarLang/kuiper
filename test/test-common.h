#ifndef __KUIPER_TEST_COMMON
#define __KUIPER_TEST_COMMON 1

#include <kuiper.h>

extern const char *progname;

/*
 * When running several tests in parallel, we may exhaust the memory
 * of the GPU. This silly wrapper will just wait and keep retrying if
 * the allocation fails.
 *
 * This should not be in this header.
 */
#include <unistd.h>
static void *kpr_wait_alloc(size_t sz, size_t len)
{
    void *ret;
    cudaError rc;
    int tries = 10;

    while ((rc = cudaMalloc(&ret, sz * len)) == cudaErrorMemoryAllocation &&
           tries-- > 0) {
        fprintf(stderr, "%s: Waiting for GPU memory...\n", progname);
        sleep(1);
    }
    MUST(rc);
    return ret;
}

#endif
