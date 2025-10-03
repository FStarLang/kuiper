#include "Kuiper_GEMM_BlockTiling1D.h"

#define stem         Kuiper_GEMM_BlockTiling1D_g_matmul_
#define et           uint32_t
#define et_lbl       u32
#define EXACT        1
#define NODYNTILE    1

#include "tiled_matmul_driver.c.inc"
