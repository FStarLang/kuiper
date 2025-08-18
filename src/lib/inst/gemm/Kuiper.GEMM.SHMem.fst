module Kuiper.GEMM.SHMem

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU {
  specialize_as_matmul_to_type_and_reprs_cpu as spec_cpu,
  specialize_as_matmul_to_type_and_reprs_gpu as spec_gpu,
  specialize_as_gemm_to_type_and_reprs_gpu as spec_gemm_gpu,
  mmcomb_gpu_tiled as mmcomb_gpu_tiled,
}
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.GEMM.SHMem

let matmul_f32_rrr tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f32 RM RM RM
let matmul_f64_rrr tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f64 RM RM RM
let matmul_u32_rrr tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u32 RM RM RM
let matmul_u64_rrr tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u64 RM RM RM


let matmul_f32_ccc tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f32 CM CM CM
let matmul_f64_ccc tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f64 CM CM CM
let matmul_u32_ccc tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u32 CM CM CM
let matmul_u64_ccc tile   = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u64 CM CM CM

let matmul_f32_tile32_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f32 RM RM RM
let matmul_f64_tile32_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f64 RM RM RM
let matmul_u32_tile32_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u32 RM RM RM
let matmul_u64_tile32_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u64 RM RM RM

let matmul_f32_tile32_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f32 CM CM CM
let matmul_f64_tile32_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f64 CM CM CM
let matmul_u32_tile32_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u32 CM CM CM
let matmul_u64_tile32_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u64 CM CM CM


let matmul_f32_tile16_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f32 RM RM RM
let matmul_f64_tile16_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f64 RM RM RM
let matmul_u32_tile16_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u32 RM RM RM
let matmul_u64_tile16_rrr = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u64 RM RM RM

let matmul_f32_tile16_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f32 CM CM CM
let matmul_f64_tile16_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f64 CM CM CM
let matmul_u32_tile16_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u32 CM CM CM
let matmul_u64_tile16_ccc = spec_cpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u64 CM CM CM




let g_matmul_f32_rrr tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f32 RM RM RM
let g_matmul_f64_rrr tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f64 RM RM RM
let g_matmul_u32_rrr tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u32 RM RM RM
let g_matmul_u64_rrr tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u64 RM RM RM


let g_matmul_f32_ccc tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f32 CM CM CM
let g_matmul_f64_ccc tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f64 CM CM CM
let g_matmul_u32_ccc tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u32 CM CM CM
let g_matmul_u64_ccc tile   = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u64 CM CM CM

let g_matmul_f32_tile32_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f32 RM RM RM
let g_matmul_f64_tile32_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f64 RM RM RM
let g_matmul_u32_tile32_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u32 RM RM RM
let g_matmul_u64_tile32_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u64 RM RM RM

let g_matmul_f32_tile32_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f32 CM CM CM
let g_matmul_f64_tile32_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f64 CM CM CM
let g_matmul_u32_tile32_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u32 CM CM CM
let g_matmul_u64_tile32_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u64 CM CM CM


let g_matmul_f32_tile16_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f32 RM RM RM
let g_matmul_f64_tile16_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f64 RM RM RM
let g_matmul_u32_tile16_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u32 RM RM RM
let g_matmul_u64_tile16_rrr = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u64 RM RM RM

let g_matmul_f32_tile16_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f32 CM CM CM
let g_matmul_f64_tile16_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f64 CM CM CM
let g_matmul_u32_tile16_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u32 CM CM CM
let g_matmul_u64_tile16_ccc = spec_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u64 CM CM CM





let g_gemm_f32_rrr tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f32 RM RM RM
let g_gemm_f64_rrr tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f64 RM RM RM
let g_gemm_u32_rrr tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u32 RM RM RM
let g_gemm_u64_rrr tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u64 RM RM RM


let g_gemm_f32_ccc tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f32 CM CM CM
let g_gemm_f64_ccc tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) f64 CM CM CM
let g_gemm_u32_ccc tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u32 CM CM CM
let g_gemm_u64_ccc tile   = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu tile) u64 CM CM CM

let g_gemm_f32_tile32_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f32 RM RM RM
let g_gemm_f64_tile32_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f64 RM RM RM
let g_gemm_u32_tile32_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u32 RM RM RM
let g_gemm_u64_tile32_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u64 RM RM RM

let g_gemm_f32_tile32_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f32 CM CM CM
let g_gemm_f64_tile32_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) f64 CM CM CM
let g_gemm_u32_tile32_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u32 CM CM CM
let g_gemm_u64_tile32_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 32sz) u64 CM CM CM


let g_gemm_f32_tile16_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f32 RM RM RM
let g_gemm_f64_tile16_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f64 RM RM RM
let g_gemm_u32_tile16_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u32 RM RM RM
let g_gemm_u64_tile16_rrr = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u64 RM RM RM

let g_gemm_f32_tile16_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f32 CM CM CM
let g_gemm_f64_tile16_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) f64 CM CM CM
let g_gemm_u32_tile16_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u32 CM CM CM
let g_gemm_u64_tile16_ccc = spec_gemm_gpu (mmcomb_gpu_tiled P.mmcomb_gpu 16sz) u64 CM CM CM
