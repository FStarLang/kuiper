#include "Klas_Softmax.h"
#include <stdint.h>

#define SCALAR       half
#define TO_DOUBLE(x) ((double)__half2float(x))
#define FROM_DOUBLE(x) __float2half((float)(x))
#define FUN(lena, a)       Klas_Softmax_softmax_f16(lena, a)
#define FUN_N(nth, lena, a) Klas_Softmax_softmax_n_f16(nth, lena, a)
#define LABEL        "softmax_f16"
#define ATOL         1e-2
#define RTOL         1e-2
#define IS_LOGSOFTMAX 0

#include "softmax_test_common.c.inc"

int main(int argc, char **argv)
{
    return run_all_tests();
}
