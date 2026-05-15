#include "Kuiper_Example_OnlineSoftmax.h"
#include "test-common.h"
#include <stdint.h>

const char *progname = "online_softmax_f32";

static void wrap_f32(uint32_t n, float *host_a, float *host_b)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof ga[0], n);
    float *gb = (float *)KPR_GPU_ALLOC(sizeof gb[0], n);
    MUST(cudaMemcpy(ga, host_a, n * sizeof ga[0], cudaMemcpyHostToDevice));
    Kuiper_Example_OnlineSoftmax__test(n, ga, gb);
    MUST(cudaMemcpy(host_b, gb, n * sizeof gb[0], cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
}

#define SCALAR        float
#define TO_DOUBLE(x)  ((double)(x))
#define FROM_DOUBLE(x) ((float)(x))
#define FUN(lena, a, b) wrap_f32(lena, a, b)
#define LABEL         "online_softmax_f32"
#define ATOL          1e-5
#define RTOL          1e-5
#define IS_LOGSOFTMAX 0
#define IS_INPLACE 0
#define HAS_VARIABLE  0

#include "softmax_test_common.c.inc"

int main(int argc, char **argv)
{
    return run_all_tests();
}
