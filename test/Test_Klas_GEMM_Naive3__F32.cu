#include "Klas_GEMM_Naive3.h"

#define stem         Klas_GEMM_Naive3_g_matmul_
#define et           float
#define et_lbl       f32
#define TOLERANCE    0.001f

#include "normal_matmul_driver.c.inc"
