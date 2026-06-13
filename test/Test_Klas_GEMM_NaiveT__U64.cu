#include "Klas_GEMM_NaiveT.h"

#define stem         Klas_GEMM_NaiveT_g_matmul_
#define et           uint64_t
#define et_lbl       u64
#define EXACT        1

#include "normal_matmul_driver.c.inc"
