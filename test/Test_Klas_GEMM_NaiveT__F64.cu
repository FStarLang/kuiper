#include "Klas_GEMM_NaiveT.h"

#define stem         Klas_GEMM_NaiveT_g_matmul_
#define et           double
#define et_lbl       f64
#define TOLERANCE    0.001f

#include "normal_matmul_driver.c.inc"
