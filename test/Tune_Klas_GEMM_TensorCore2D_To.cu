#include "Klas_GEMM_TensorCore2D_To.h"

// #define stem          Klas_GEMM_TensorCore2D_To_g_gemm_
#define et             __nv_bfloat16
#define et_is_bf16     1
#define cet            __nv_bfloat16
#define aet            float
#define aet_is_float   1
#define separate_output 1
// #define et_lbl        // bf16_f32_bf16
// #define tile_sizes    _128x128x32
// #define tc_tile_sizes _16x16x16
// #define regch_sizes   _8x8
#define GEMM_ALPHA     0.7
#define GEMM_BETA      0.3
#define TOLERANCE      0.02f
#define PREARGS

#include "tensor_core_gemm_alpha_beta_1_driver.c.inc"
