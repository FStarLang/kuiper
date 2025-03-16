#include "Kuiper_MatMul_SHMem.h"

#define stem         Kuiper_MatMul_SHMem_g_matmul_
#define et           double
#define et_lbl       f64
#define PRIet        "f"
#define TOLERANCE    0.001f

#include "tiled_matmul_driver.c.inc"
