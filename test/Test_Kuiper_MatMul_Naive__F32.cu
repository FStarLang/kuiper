#include "Kuiper_MatMul_Naive.h"

#define stem         Kuiper_MatMul_Naive_g_matmul_
#define et           float
#define et_lbl       f32
#define PRIet        "f"
#define TOLERANCE    0.001f

#include "normal_matmul_driver.c.inc"
