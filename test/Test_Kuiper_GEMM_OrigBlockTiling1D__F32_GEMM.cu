#include "Kuiper_GEMM_OrigBlockTiling1D.h"

#define stem         Kuiper_GEMM_OrigBlockTiling1D_g_gemm_
#define et           float
#define et_lbl       f32
#define GEMM_ALPHA   0.7
#define GEMM_BETA    0.3
#define TOLERANCE    0.001f
#define NODYNTILE    1

#include "block_tiled1d_gemm_driver.c.inc"
