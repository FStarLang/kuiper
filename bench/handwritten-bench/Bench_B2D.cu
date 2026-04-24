#include <stdint.h>
#include "BlockTiling2D.h"

#define stem         blocktiling2d_host
#define et           float
#define et_lbl
#define tile_sizes
#define regch_sizes
#define layouts
#define GEMM_ALPHA   0.7
#define GEMM_BETA    0.3
#define TOLERANCE    0.001f
#define NODYNTILE    1
#define PREARGS                 //128,128,32,

#include "block_tiled2d_gemm_driver.c.inc"
