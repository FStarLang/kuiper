#include "Klas_GEMM_Tiled.h"

#define stem         Klas_GEMM_Tiled_g_matmul_
#define et           uint32_t
#define et_lbl       u32
#define EXACT        1

#include "tiled_matmul_driver.c.inc"
