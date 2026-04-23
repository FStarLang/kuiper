#include "Klas_LogSoftmax.h"
#include <stdint.h>

#define SCALAR       double
#define TO_DOUBLE(x) ((double)(x))
#define FROM_DOUBLE(x) ((double)(x))
#define FUN(lena, a)       Klas_LogSoftmax_log_softmax_f64(lena, a)
#define FUN_N(nth, lena, a) Klas_LogSoftmax_log_softmax_n_f64(nth, lena, a)
#define LABEL        "logsoftmax_f64"
#define ATOL         1e-10
#define RTOL         1e-10
#define IS_LOGSOFTMAX 1

#include "softmax_test_common.c.inc"

int main(int argc, char **argv)
{
    return run_all_tests();
}
