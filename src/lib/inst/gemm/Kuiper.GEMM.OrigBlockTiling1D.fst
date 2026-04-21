module Kuiper.GEMM.OrigBlockTiling1D

#lang-pulse
open Kuiper
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
open Kuiper.GEMM.OrigBlockTiling1D.Inst
module MS = Kuiper.Spec.GEMM

let matmul_f32_tiles64x8_8x64_rc8_rrr =
  spec 64sz 64sz 8sz 8sz f32 (fun _ n -> n) (fun _ r -> r) RM RM RM

// This combination of tile sizes is forbidden!
//  The generated code contains dynamic guards that immediately fail.
// let matmul_f32_tiles32x32_32x32_rc32_rrr = spec_cpu (mmcomb_gpu_block_tiled1d P.mmcomb_gpu 32sz 32sz 32sz 32sz) f32 RM RM RM

let g_gemm_f32_tiles64x8_8x64_rc8_rrr alpha beta =
  to_real_ok alpha;
  to_real_ok beta;
  spec 64sz 64sz 8sz 8sz f32
    (MS.lincomb alpha beta) (MS.lincomb (to_real alpha) (to_real beta))
     RM RM RM
