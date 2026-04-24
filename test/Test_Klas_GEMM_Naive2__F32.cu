#include "Klas_GEMM_Naive2.h"

#define stem         Klas_GEMM_Naive2_g_matmul_
#define et           float
#define et_lbl       f32
#define TOLERANCE    0.001f

#include "normal_matmul_driver.c.inc"
