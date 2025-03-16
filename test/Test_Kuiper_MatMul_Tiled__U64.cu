#include "Kuiper_MatMul_Tiled.h"

#define stem         Kuiper_MatMul_Tiled_g_matmul_
#define et           uint64_t
#define et_lbl       u64
#define PRIet        PRIu64
#define EXACT        1

#include "tiled_matmul_driver.c.inc"
