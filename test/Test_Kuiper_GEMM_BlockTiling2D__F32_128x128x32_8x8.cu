#include "Kuiper_GEMM_BlockTiling2D.h"

#define stem         Kuiper_GEMM_BlockTiling2D_g_gemm_
#define et           float
#define et_lbl       f32
#ifndef tile_sizes
#define tile_sizes   _128x128x32
#endif
#ifndef regch_sizes
#define regch_sizes  _8x8
#endif
#define layouts      _cr
#define GEMM_ALPHA   0.7
#define GEMM_BETA    0.3
#define TOLERANCE    0.001f
#define NODYNTILE    1
#define PREARGS      //128,128,32,

#include "block_tiled2d_gemm_driver.c.inc"
