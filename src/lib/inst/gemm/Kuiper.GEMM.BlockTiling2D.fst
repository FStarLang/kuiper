module Kuiper.GEMM.BlockTiling2D

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU {
  specialize_as_matmul_to_type_and_reprs_cpu as spec_cpu,
  specialize_as_gemm_to_type_and_reprs_gpu as spec_gemm_gpu,
  mmcomb_gpu_block_tiled2d
}
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.GEMM.BlockTiling2D

let matmul_f32_64x8x64_8x8_rrr = spec_cpu (mmcomb_gpu_block_tiled2d P.mmcomb_gpu 64sz 64sz 8sz 8sz 8sz) f32 RM RM RM
let matmul_f32_32x32x32_32x8_rrr = spec_cpu (mmcomb_gpu_block_tiled2d P.mmcomb_gpu 32sz 32sz 32sz 32sz 8sz) f32 RM RM RM

let g_gemm_f32_64x8x64_8x8_rrr = spec_gemm_gpu (mmcomb_gpu_block_tiled2d P.mmcomb_gpu 64sz 64sz 8sz 8sz 8sz) f32 RM RM RM
let g_gemm_f32_128x8x128_8x8_rrr = spec_gemm_gpu (mmcomb_gpu_block_tiled2d P.mmcomb_gpu 128sz 128sz 8sz 8sz 8sz) f32 RM RM RM