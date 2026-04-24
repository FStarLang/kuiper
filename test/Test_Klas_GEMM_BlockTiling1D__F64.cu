#include "Klas_GEMM_BlockTiling1D.h"

#define stem         Klas_GEMM_BlockTiling1D_g_matmul_
#define et           double
#define et_lbl       f64
#define TOLERANCE    0.001f
#define NODYNTILE    1

#include "tiled_matmul_driver.c.inc"
