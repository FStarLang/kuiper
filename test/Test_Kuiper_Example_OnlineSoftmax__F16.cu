#include "Kuiper_Example_OnlineSoftmax.h"
#include "test-common.h"
#include <stdint.h>

const char *progname = "online_softmax_f16";

static void wrap_f16(uint32_t n, half *host_a, half *host_b)
{
    half *ga = (half *)KPR_GPU_ALLOC(sizeof ga[0], n);
    half *gb = (half *)KPR_GPU_ALLOC(sizeof gb[0], n);
    MUST(cudaMemcpy(ga, host_a, n * sizeof ga[0], cudaMemcpyHostToDevice));
    Kuiper_Example_OnlineSoftmax__testh(n, ga, gb);
    MUST(cudaMemcpy(host_b, gb, n * sizeof gb[0], cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    MUST(cudaFree(gb));
}

#define SCALAR        half
#define TO_DOUBLE(x)  ((double)__half2float(x))
#define FROM_DOUBLE(x) __float2half((float)(x))
#define FUN(lena, a, b) wrap_f16(lena, a, b)
#define LABEL         "online_softmax_f16"
#define ATOL          1e-2
#define RTOL          1e-2
#define IS_LOGSOFTMAX 0
#define IS_INPLACE 0
#define HAS_VARIABLE  0

#include "softmax_test_common.c.inc"

int main(int argc, char **argv)
{
    return run_all_tests();
}
