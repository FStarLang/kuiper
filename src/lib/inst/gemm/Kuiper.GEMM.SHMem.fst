module Kuiper.GEMM.SHMem

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMCPU {
  specialize_tiled_approx_gpu as spec_approx_gpu,
  specialize_tiled_approx_cpu as spec_approx_cpu,
  specialize_tiled_approx_gemm_gpu as spec_approx_gemm
}
open Kuiper.Poly.GEMMGPU.Type { valid_tile }
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.GEMM.SHMem

(* CPU approximate matmul - dynamically-chosen tile *)

let matmul_f32_rrr tile = spec_approx_cpu P.mmcomb_gpu_approx tile f32 RM RM RM
let matmul_f64_rrr tile = spec_approx_cpu P.mmcomb_gpu_approx tile f64 RM RM RM
let matmul_u32_rrr tile = spec_approx_cpu P.mmcomb_gpu_approx tile u32 RM RM RM
let matmul_u64_rrr tile = spec_approx_cpu P.mmcomb_gpu_approx tile u64 RM RM RM

let matmul_f32_ccc tile = spec_approx_cpu P.mmcomb_gpu_approx tile f32 CM CM CM
let matmul_f64_ccc tile = spec_approx_cpu P.mmcomb_gpu_approx tile f64 CM CM CM
let matmul_u32_ccc tile = spec_approx_cpu P.mmcomb_gpu_approx tile u32 CM CM CM
let matmul_u64_ccc tile = spec_approx_cpu P.mmcomb_gpu_approx tile u64 CM CM CM

(* CPU approximate matmul - tile=32 *)

let matmul_f32_tile32_rrr = spec_approx_cpu P.mmcomb_gpu_approx 32sz f32 RM RM RM
let matmul_f64_tile32_rrr = spec_approx_cpu P.mmcomb_gpu_approx 32sz f64 RM RM RM
let matmul_u32_tile32_rrr = spec_approx_cpu P.mmcomb_gpu_approx 32sz u32 RM RM RM
let matmul_u64_tile32_rrr = spec_approx_cpu P.mmcomb_gpu_approx 32sz u64 RM RM RM

let matmul_f32_tile32_ccc = spec_approx_cpu P.mmcomb_gpu_approx 32sz f32 CM CM CM
let matmul_f64_tile32_ccc = spec_approx_cpu P.mmcomb_gpu_approx 32sz f64 CM CM CM
let matmul_u32_tile32_ccc = spec_approx_cpu P.mmcomb_gpu_approx 32sz u32 CM CM CM
let matmul_u64_tile32_ccc = spec_approx_cpu P.mmcomb_gpu_approx 32sz u64 CM CM CM

(* CPU approximate matmul - tile=16 *)

let matmul_f32_tile16_rrr = spec_approx_cpu P.mmcomb_gpu_approx 16sz f32 RM RM RM
let matmul_f64_tile16_rrr = spec_approx_cpu P.mmcomb_gpu_approx 16sz f64 RM RM RM
let matmul_u32_tile16_rrr = spec_approx_cpu P.mmcomb_gpu_approx 16sz u32 RM RM RM
let matmul_u64_tile16_rrr = spec_approx_cpu P.mmcomb_gpu_approx 16sz u64 RM RM RM

let matmul_f32_tile16_ccc = spec_approx_cpu P.mmcomb_gpu_approx 16sz f32 CM CM CM
let matmul_f64_tile16_ccc = spec_approx_cpu P.mmcomb_gpu_approx 16sz f64 CM CM CM
let matmul_u32_tile16_ccc = spec_approx_cpu P.mmcomb_gpu_approx 16sz u32 CM CM CM
let matmul_u64_tile16_ccc = spec_approx_cpu P.mmcomb_gpu_approx 16sz u64 CM CM CM

(* GPU-side approximate matmul - dynamically-chosen tile *)

let g_matmul_f32_rrr tile = spec_approx_gpu P.mmcomb_gpu_approx tile f32 RM RM RM
let g_matmul_f64_rrr tile = spec_approx_gpu P.mmcomb_gpu_approx tile f64 RM RM RM
let g_matmul_u32_rrr tile = spec_approx_gpu P.mmcomb_gpu_approx tile u32 RM RM RM
let g_matmul_u64_rrr tile = spec_approx_gpu P.mmcomb_gpu_approx tile u64 RM RM RM

let g_matmul_f32_ccc tile = spec_approx_gpu P.mmcomb_gpu_approx tile f32 CM CM CM
let g_matmul_f64_ccc tile = spec_approx_gpu P.mmcomb_gpu_approx tile f64 CM CM CM
let g_matmul_u32_ccc tile = spec_approx_gpu P.mmcomb_gpu_approx tile u32 CM CM CM
let g_matmul_u64_ccc tile = spec_approx_gpu P.mmcomb_gpu_approx tile u64 CM CM CM

(* GPU-side approximate matmul - tile=32 *)

let g_matmul_f32_tile32_rrr = spec_approx_gpu P.mmcomb_gpu_approx 32sz f32 RM RM RM
let g_matmul_f64_tile32_rrr = spec_approx_gpu P.mmcomb_gpu_approx 32sz f64 RM RM RM
let g_matmul_u32_tile32_rrr = spec_approx_gpu P.mmcomb_gpu_approx 32sz u32 RM RM RM
let g_matmul_u64_tile32_rrr = spec_approx_gpu P.mmcomb_gpu_approx 32sz u64 RM RM RM

let g_matmul_f32_tile32_ccc = spec_approx_gpu P.mmcomb_gpu_approx 32sz f32 CM CM CM
let g_matmul_f64_tile32_ccc = spec_approx_gpu P.mmcomb_gpu_approx 32sz f64 CM CM CM
let g_matmul_u32_tile32_ccc = spec_approx_gpu P.mmcomb_gpu_approx 32sz u32 CM CM CM
let g_matmul_u64_tile32_ccc = spec_approx_gpu P.mmcomb_gpu_approx 32sz u64 CM CM CM

(* GPU-side approximate matmul - tile=16 *)

let g_matmul_f32_tile16_rrr = spec_approx_gpu P.mmcomb_gpu_approx 16sz f32 RM RM RM
let g_matmul_f64_tile16_rrr = spec_approx_gpu P.mmcomb_gpu_approx 16sz f64 RM RM RM
let g_matmul_u32_tile16_rrr = spec_approx_gpu P.mmcomb_gpu_approx 16sz u32 RM RM RM
let g_matmul_u64_tile16_rrr = spec_approx_gpu P.mmcomb_gpu_approx 16sz u64 RM RM RM

let g_matmul_f32_tile16_ccc = spec_approx_gpu P.mmcomb_gpu_approx 16sz f32 CM CM CM
let g_matmul_f64_tile16_ccc = spec_approx_gpu P.mmcomb_gpu_approx 16sz f64 CM CM CM
let g_matmul_u32_tile16_ccc = spec_approx_gpu P.mmcomb_gpu_approx 16sz u32 CM CM CM
let g_matmul_u64_tile16_ccc = spec_approx_gpu P.mmcomb_gpu_approx 16sz u64 CM CM CM

(* GPU-side approximate GEMM - dynamically-chosen tile *)

let g_gemm_f32_rrr tile = spec_approx_gemm P.mmcomb_gpu_approx tile f32 RM RM RM
let g_gemm_f64_rrr tile = spec_approx_gemm P.mmcomb_gpu_approx tile f64 RM RM RM
let g_gemm_u32_rrr tile = spec_approx_gemm P.mmcomb_gpu_approx tile u32 RM RM RM
let g_gemm_u64_rrr tile = spec_approx_gemm P.mmcomb_gpu_approx tile u64 RM RM RM

let g_gemm_f32_ccc tile = spec_approx_gemm P.mmcomb_gpu_approx tile f32 CM CM CM
let g_gemm_f64_ccc tile = spec_approx_gemm P.mmcomb_gpu_approx tile f64 CM CM CM
let g_gemm_u32_ccc tile = spec_approx_gemm P.mmcomb_gpu_approx tile u32 CM CM CM
let g_gemm_u64_ccc tile = spec_approx_gemm P.mmcomb_gpu_approx tile u64 CM CM CM

(* GPU-side approximate GEMM - tile=32 *)

let g_gemm_f32_tile32_rrr = spec_approx_gemm P.mmcomb_gpu_approx 32sz f32 RM RM RM
let g_gemm_f64_tile32_rrr = spec_approx_gemm P.mmcomb_gpu_approx 32sz f64 RM RM RM
let g_gemm_u32_tile32_rrr = spec_approx_gemm P.mmcomb_gpu_approx 32sz u32 RM RM RM
let g_gemm_u64_tile32_rrr = spec_approx_gemm P.mmcomb_gpu_approx 32sz u64 RM RM RM

let g_gemm_f32_tile32_ccc = spec_approx_gemm P.mmcomb_gpu_approx 32sz f32 CM CM CM
let g_gemm_f64_tile32_ccc = spec_approx_gemm P.mmcomb_gpu_approx 32sz f64 CM CM CM
let g_gemm_u32_tile32_ccc = spec_approx_gemm P.mmcomb_gpu_approx 32sz u32 CM CM CM
let g_gemm_u64_tile32_ccc = spec_approx_gemm P.mmcomb_gpu_approx 32sz u64 CM CM CM

(* GPU-side approximate GEMM - tile=16 *)

let g_gemm_f32_tile16_rrr = spec_approx_gemm P.mmcomb_gpu_approx 16sz f32 RM RM RM
let g_gemm_f64_tile16_rrr = spec_approx_gemm P.mmcomb_gpu_approx 16sz f64 RM RM RM
let g_gemm_u32_tile16_rrr = spec_approx_gemm P.mmcomb_gpu_approx 16sz u32 RM RM RM
let g_gemm_u64_tile16_rrr = spec_approx_gemm P.mmcomb_gpu_approx 16sz u64 RM RM RM

let g_gemm_f32_tile16_ccc = spec_approx_gemm P.mmcomb_gpu_approx 16sz f32 CM CM CM
let g_gemm_f64_tile16_ccc = spec_approx_gemm P.mmcomb_gpu_approx 16sz f64 CM CM CM
let g_gemm_u32_tile16_ccc = spec_approx_gemm P.mmcomb_gpu_approx 16sz u32 CM CM CM
let g_gemm_u64_tile16_ccc = spec_approx_gemm P.mmcomb_gpu_approx 16sz u64 CM CM CM
