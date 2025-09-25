#include "Kuiper_GEMM_TensorCore.h"

#define stem          Kuiper_GEMM_TensorCore_g_gemm_
#define et            half
#define et_is_half    1
#define et_lbl        f16_f16
#define tile_sizes    _64x64x64
#define tc_tile_sizes _32x8x16
#define layouts       _rrr
#define PRIet         "f"
#define GEMM_ALPHA    1.0
#define GEMM_BETA     1.0
#define TOLERANCE     0.001f
// #define NODYNTILE     1

#include "tensor_core_gemm_alpha_beta_1_driver.c.inc"
