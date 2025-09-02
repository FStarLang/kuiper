module Kuiper.GEMM.OrigBlockTiling1D

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU {
  specialize_as_matmul_to_type_and_reprs_cpu as spec_cpu,
  specialize_as_gemm_to_type_and_reprs_gpu as spec_gemm_gpu,
  mmcomb_gpu_block_tiled1d
}
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.GEMM.OrigBlockTiling1D

let matmul_f32_tiles64x8_8x64_rc8_rrr = spec_cpu (mmcomb_gpu_block_tiled1d P.mmcomb_gpu 64sz 64sz 8sz 8sz) f32 RM RM RM
// This combination of tile sizes is forbidden!
//  The generated code contains dynamic guards that immediately fail.
// let matmul_f32_tiles32x32_32x32_rc32_rrr = spec_cpu (mmcomb_gpu_block_tiled1d P.mmcomb_gpu 32sz 32sz 32sz 32sz) f32 RM RM RM

let g_gemm_f32_tiles64x8_8x64_rc8_rrr = spec_gemm_gpu (mmcomb_gpu_block_tiled1d P.mmcomb_gpu 64sz 64sz 8sz 8sz) f32 RM RM RM
