#include "Klas_GEMM_Tiled.h"

#define stem         Klas_GEMM_Tiled_g_matmul_
#define et           uint64_t
#define et_lbl       u64
#define EXACT        1

#include "tiled_matmul_driver.c.inc"
