module Kuiper.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm }
open Kuiper.GEMM.BlockTiling2D.Inst { spec }

// Dynamic parameter version. Only admitted since otherwise we
// need to repeat all requirements here. We only use this for some
// quick tuning, so it\'s not a big deal.
let g_gemm_f32_8x8_cr bm bn bk =
  admit();
  spec bm bn bk (cm _ _) (rm _ _)
    8sz 8sz f32

let g_gemm_f32_64x64x8_8x8_cr = spec 64sz 64sz 8sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x64x8_8x16_cr = spec 64sz 64sz 8sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x64x8_16x8_cr = spec 64sz 64sz 8sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x64x8_16x16_cr = spec 64sz 64sz 8sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x64x16_8x8_cr = spec 64sz 64sz 16sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x64x16_8x16_cr = spec 64sz 64sz 16sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x64x16_16x8_cr = spec 64sz 64sz 16sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x64x16_16x16_cr = spec 64sz 64sz 16sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x64x32_8x8_cr = spec 64sz 64sz 32sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x64x32_8x16_cr = spec 64sz 64sz 32sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x64x32_16x8_cr = spec 64sz 64sz 32sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x64x32_16x16_cr = spec 64sz 64sz 32sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x64x64_8x8_cr = spec 64sz 64sz 64sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x64x64_8x16_cr = spec 64sz 64sz 64sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x64x64_16x8_cr = spec 64sz 64sz 64sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x64x64_16x16_cr = spec 64sz 64sz 64sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x128x8_8x8_cr = spec 64sz 128sz 8sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x128x8_8x16_cr = spec 64sz 128sz 8sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x128x8_16x8_cr = spec 64sz 128sz 8sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x128x8_16x16_cr = spec 64sz 128sz 8sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x128x16_8x8_cr = spec 64sz 128sz 16sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x128x16_8x16_cr = spec 64sz 128sz 16sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x128x16_16x8_cr = spec 64sz 128sz 16sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x128x16_16x16_cr = spec 64sz 128sz 16sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x128x32_8x8_cr = spec 64sz 128sz 32sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x128x32_8x16_cr = spec 64sz 128sz 32sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x128x32_16x8_cr = spec 64sz 128sz 32sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x128x32_16x16_cr = spec 64sz 128sz 32sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_64x128x64_8x8_cr = spec 64sz 128sz 64sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_64x128x64_8x16_cr = spec 64sz 128sz 64sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_64x128x64_16x8_cr = spec 64sz 128sz 64sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_64x128x64_16x16_cr = spec 64sz 128sz 64sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x64x8_8x8_cr = spec 128sz 64sz 8sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x64x8_8x16_cr = spec 128sz 64sz 8sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x64x8_16x8_cr = spec 128sz 64sz 8sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x64x8_16x16_cr = spec 128sz 64sz 8sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x64x16_8x8_cr = spec 128sz 64sz 16sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x64x16_8x16_cr = spec 128sz 64sz 16sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x64x16_16x8_cr = spec 128sz 64sz 16sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x64x16_16x16_cr = spec 128sz 64sz 16sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x64x32_8x8_cr = spec 128sz 64sz 32sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x64x32_8x16_cr = spec 128sz 64sz 32sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x64x32_16x8_cr = spec 128sz 64sz 32sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x64x32_16x16_cr = spec 128sz 64sz 32sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x64x64_8x8_cr = spec 128sz 64sz 64sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x64x64_8x16_cr = spec 128sz 64sz 64sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x64x64_16x8_cr = spec 128sz 64sz 64sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x64x64_16x16_cr = spec 128sz 64sz 64sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x128x8_8x8_cr = spec 128sz 128sz 8sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x128x8_8x16_cr = spec 128sz 128sz 8sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x128x8_16x8_cr = spec 128sz 128sz 8sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x128x8_16x16_cr = spec 128sz 128sz 8sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x128x16_8x8_cr = spec 128sz 128sz 16sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x128x16_8x16_cr = spec 128sz 128sz 16sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x128x16_16x8_cr = spec 128sz 128sz 16sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x128x16_16x16_cr = spec 128sz 128sz 16sz (cm _ _) (rm _ _) 16sz 16sz f32
let g_gemm_f32_128x128x32_8x8_cr = spec 128sz 128sz 32sz (cm _ _) (rm _ _) 8sz 8sz f32
let g_gemm_f32_128x128x32_8x16_cr = spec 128sz 128sz 32sz (cm _ _) (rm _ _) 8sz 16sz f32
let g_gemm_f32_128x128x32_16x8_cr = spec 128sz 128sz 32sz (cm _ _) (rm _ _) 16sz 8sz f32
let g_gemm_f32_128x128x32_16x16_cr = spec 128sz 128sz 32sz (cm _ _) (rm _ _) 16sz 16sz f32
