#include "Klas_GEMM_SHMem.h"

#define stem         Klas_GEMM_SHMem_g_matmul_
#define et           double
#define et_lbl       f64
#define TOLERANCE    0.001f

#include "tiled_matmul_driver.c.inc"
