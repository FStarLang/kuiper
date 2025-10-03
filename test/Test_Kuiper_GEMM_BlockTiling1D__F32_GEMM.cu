#include "Kuiper_GEMM_BlockTiling1D.h"

#define stem         Kuiper_GEMM_BlockTiling1D_g_gemm_
#define et           float
#define et_lbl       f32
#define GEMM_ALPHA   0.7
#define GEMM_BETA    0.3
#define TOLERANCE    0.001f
#define NODYNTILE    1

#include "tiled_gemm_driver.c.inc"
