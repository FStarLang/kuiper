#include "Klas_GEMM_SHMem.h"

#define stem         Klas_GEMM_SHMem_g_matmul_
#define et           float
#define et_lbl       f32
#define TOLERANCE    0.001f

#include "tiled_matmul_driver.c.inc"
