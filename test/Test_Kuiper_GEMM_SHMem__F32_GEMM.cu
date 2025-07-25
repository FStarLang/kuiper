#include "Kuiper_GEMM_SHMem.h"

#define stem         Kuiper_GEMM_SHMem_g_gemm_
#define et           float
#define et_lbl       f32
#define PRIet        "f"
#define GEMM_ALPHA   0.7
#define GEMM_BETA    0.3
#define TOLERANCE    0.001f

#include "tiled_gemm_driver.c.inc"
