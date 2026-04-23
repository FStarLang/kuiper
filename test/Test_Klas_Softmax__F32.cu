#include "Klas_Softmax.h"
#include <stdint.h>

#define SCALAR       float
#define TO_DOUBLE(x) ((double)(x))
#define FROM_DOUBLE(x) ((float)(x))
#define FUN(lena, a)       Klas_Softmax_softmax_f32(lena, a)
#define FUN_N(nth, lena, a) Klas_Softmax_softmax_n_f32(nth, lena, a)
#define LABEL        "softmax_f32"
#define ATOL         1e-5
#define RTOL         1e-5
#define IS_LOGSOFTMAX 0

#include "softmax_test_common.c.inc"

int main(int argc, char **argv)
{
    return run_all_tests();
}
