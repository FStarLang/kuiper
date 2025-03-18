#include "Kuiper_GEMM_Tiled.h"

#define stem         Kuiper_GEMM_Tiled_g_matmul_
#define et           float
#define et_lbl       f32
#define PRIet        "f"
#define TOLERANCE    0.001f

#include "tiled_matmul_driver.c.inc"
