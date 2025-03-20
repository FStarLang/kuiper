#include "Kuiper_GEMM_Naive.h"

#define stem         Kuiper_GEMM_Naive_g_matmul_
#define et           double
#define et_lbl       f64
#define PRIet        "f"
#define TOLERANCE    0.001f

#include "normal_matmul_driver.c.inc"
