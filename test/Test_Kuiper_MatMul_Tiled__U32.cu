#include "Kuiper_MatMul_Tiled.h"

#define stem         Kuiper_MatMul_Tiled_g_matmul_
#define et           uint32_t
#define et_lbl       u32
#define PRIet        PRIu32
#define EXACT        1

#include "tiled_matmul_driver.c.inc"
