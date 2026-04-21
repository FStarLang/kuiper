#include "Klas_GEMM_TensorCore2D.h"

#define stem          Klas_GEMM_TensorCore2D_g_gemm_
#define et            half
#define et_is_half    1
#define et_lbl        f16_f16
#define tile_sizes    _64x64x64
#define tc_tile_sizes _16x16x16
#define regch_sizes   _2x2
#define GEMM_ALPHA    1.0
#define GEMM_BETA     1.0
#define TOLERANCE 0.02f
#define PREARGS

#include "tensor_core_gemm_alpha_beta_1_driver.c.inc"
