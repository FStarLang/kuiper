#include "Kuiper_GEMM_BlockTiling1D.h"

#define stem         Kuiper_GEMM_BlockTiling1D_g_matmul_
#define et           float
#define et_lbl       f32
#define PRIet        "f"
#define TOLERANCE    0.001f
#define NODYNTILE    1

#include "tiled_matmul_driver.c.inc"
