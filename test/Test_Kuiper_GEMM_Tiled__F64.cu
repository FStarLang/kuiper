#include "Kuiper_GEMM_Tiled.h"

#define stem         Kuiper_GEMM_Tiled_g_matmul_
#define et           double
#define et_lbl       f64
#define TOLERANCE    0.001f

#include "tiled_matmul_driver.c.inc"
