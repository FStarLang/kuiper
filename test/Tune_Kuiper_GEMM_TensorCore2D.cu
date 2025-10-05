#include "Kuiper_GEMM_TensorCore2D.h"

// #define stem         Kuiper_GEMM_TensorCore2D_g_gemm_
#define et           half
#define et_is_half    1
#define cet          half
// #define et_lbl       // f16_f16
// #define tile_sizes   _128x128x32
// #define tc_tile_sizes _16x16x16
// #define regch_sizes  _8x8
#define GEMM_ALPHA   0.7
#define GEMM_BETA    0.3
#define TOLERANCE 0.20f
#define PREARGS                 //128,128,32,

#include "tensor_core_gemm_alpha_beta_1_driver.c.inc"
