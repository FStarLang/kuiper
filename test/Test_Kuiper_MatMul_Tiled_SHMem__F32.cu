#include "Kuiper_MatMul_Tiled_SHMem.h"

#define stem         Kuiper_MatMul_Tiled_SHMem_g_matmul_
#define et           float
#define et_lbl       f32
#define PRIet        "f"
#define TOLERANCE    0.001f

#include "tiled_matmul_driver.c.inc"
