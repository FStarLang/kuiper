#include "Kuiper_GEMM_TensorCorePiped.h"

#define stem          Kuiper_GEMM_TensorCorePiped_align_g_gemm_
#define et            half
#define et_is_half    1
#define et_lbl        f16_f16
#define tile_sizes    _64x64x32
#define tc_tile_sizes _16x16x16
#define regch_sizes   _4x4
#define GEMM_ALPHA    1.0
#define GEMM_BETA     1.0
#define TOLERANCE 0.20f
#define PREARGS

#include "tensor_core_gemm_alpha_beta_1_driver.c.inc"
