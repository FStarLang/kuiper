#include "Kuiper_GEMM_TensorCore2D.h"

#define stem          Kuiper_GEMM_TensorCore2D_g_gemm_
#define et            half
#define et_is_half    1
#define et_lbl        f16_f16
#define tile_sizes    _128x128x32
#define tc_tile_sizes _16x16x16_2x2
#define layouts
#define PRIet         "f"
#define GEMM_ALPHA    1.0
#define GEMM_BETA     1.0
#define TOLERANCE     0.25f
// #define NODYNTILE     1
#define PREARGS       //128,128,32,

#include "tensor_core_gemm_alpha_beta_1_driver.c.inc"
