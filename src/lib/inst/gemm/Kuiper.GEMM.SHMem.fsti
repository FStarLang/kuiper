module Kuiper.GEMM.SHMem

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMCPU
open Kuiper.Poly.GEMMGPU.Type { valid_tile }
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }

inline_for_extraction noextract
let size_req tile : size_req_t =
  fun rows shared cols ->
    (rows / tile) * (cols / tile) <= max_blocks /\
    tile * tile <= max_threads

(* CPU approximate matmul - dynamically-chosen tile *)

val matmul_f32_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty f32 (size_req tile) RM RM RM
val matmul_f64_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty f64 (size_req tile) RM RM RM
val matmul_u32_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty u32 (size_req tile) RM RM RM
val matmul_u64_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty u64 (size_req tile) RM RM RM

val matmul_f32_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty f32 (size_req tile) CM CM CM
val matmul_f64_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty f64 (size_req tile) CM CM CM
val matmul_u32_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty u32 (size_req tile) CM CM CM
val matmul_u64_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_approx_ty u64 (size_req tile) CM CM CM

(* CPU approximate matmul - tile=32 *)

val matmul_f32_tile32_rrr : fixed_repr_matmul_cpu_approx_ty f32 (size_req 32) RM RM RM
val matmul_f64_tile32_rrr : fixed_repr_matmul_cpu_approx_ty f64 (size_req 32) RM RM RM
val matmul_u32_tile32_rrr : fixed_repr_matmul_cpu_approx_ty u32 (size_req 32) RM RM RM
val matmul_u64_tile32_rrr : fixed_repr_matmul_cpu_approx_ty u64 (size_req 32) RM RM RM

val matmul_f32_tile32_ccc : fixed_repr_matmul_cpu_approx_ty f32 (size_req 32) CM CM CM
val matmul_f64_tile32_ccc : fixed_repr_matmul_cpu_approx_ty f64 (size_req 32) CM CM CM
val matmul_u32_tile32_ccc : fixed_repr_matmul_cpu_approx_ty u32 (size_req 32) CM CM CM
val matmul_u64_tile32_ccc : fixed_repr_matmul_cpu_approx_ty u64 (size_req 32) CM CM CM

(* CPU approximate matmul - tile=16 *)

val matmul_f32_tile16_rrr : fixed_repr_matmul_cpu_approx_ty f32 (size_req 16) RM RM RM
val matmul_f64_tile16_rrr : fixed_repr_matmul_cpu_approx_ty f64 (size_req 16) RM RM RM
val matmul_u32_tile16_rrr : fixed_repr_matmul_cpu_approx_ty u32 (size_req 16) RM RM RM
val matmul_u64_tile16_rrr : fixed_repr_matmul_cpu_approx_ty u64 (size_req 16) RM RM RM

val matmul_f32_tile16_ccc : fixed_repr_matmul_cpu_approx_ty f32 (size_req 16) CM CM CM
val matmul_f64_tile16_ccc : fixed_repr_matmul_cpu_approx_ty f64 (size_req 16) CM CM CM
val matmul_u32_tile16_ccc : fixed_repr_matmul_cpu_approx_ty u32 (size_req 16) CM CM CM
val matmul_u64_tile16_ccc : fixed_repr_matmul_cpu_approx_ty u64 (size_req 16) CM CM CM

(* GPU-side approximate matmul - dynamically-chosen tile *)

val g_matmul_f32_rrr (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty f32 (size_req tile) RM RM RM
val g_matmul_f64_rrr (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty f64 (size_req tile) RM RM RM
val g_matmul_u32_rrr (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty u32 (size_req tile) RM RM RM
val g_matmul_u64_rrr (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty u64 (size_req tile) RM RM RM

val g_matmul_f32_ccc (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty f32 (size_req tile) CM CM CM
val g_matmul_f64_ccc (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty f64 (size_req tile) CM CM CM
val g_matmul_u32_ccc (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty u32 (size_req tile) CM CM CM
val g_matmul_u64_ccc (tile : valid_tile) : fixed_repr_mmcomb_gpu_approx_ty u64 (size_req tile) CM CM CM

(* GPU-side approximate matmul - tile=32 *)

val g_matmul_f32_tile32_rrr : fixed_repr_mmcomb_gpu_approx_ty f32 (size_req 32) RM RM RM
val g_matmul_f64_tile32_rrr : fixed_repr_mmcomb_gpu_approx_ty f64 (size_req 32) RM RM RM
val g_matmul_u32_tile32_rrr : fixed_repr_mmcomb_gpu_approx_ty u32 (size_req 32) RM RM RM
val g_matmul_u64_tile32_rrr : fixed_repr_mmcomb_gpu_approx_ty u64 (size_req 32) RM RM RM

val g_matmul_f32_tile32_ccc : fixed_repr_mmcomb_gpu_approx_ty f32 (size_req 32) CM CM CM
val g_matmul_f64_tile32_ccc : fixed_repr_mmcomb_gpu_approx_ty f64 (size_req 32) CM CM CM
val g_matmul_u32_tile32_ccc : fixed_repr_mmcomb_gpu_approx_ty u32 (size_req 32) CM CM CM
val g_matmul_u64_tile32_ccc : fixed_repr_mmcomb_gpu_approx_ty u64 (size_req 32) CM CM CM

(* GPU-side approximate matmul - tile=16 *)

val g_matmul_f32_tile16_rrr : fixed_repr_mmcomb_gpu_approx_ty f32 (size_req 16) RM RM RM
val g_matmul_f64_tile16_rrr : fixed_repr_mmcomb_gpu_approx_ty f64 (size_req 16) RM RM RM
val g_matmul_u32_tile16_rrr : fixed_repr_mmcomb_gpu_approx_ty u32 (size_req 16) RM RM RM
val g_matmul_u64_tile16_rrr : fixed_repr_mmcomb_gpu_approx_ty u64 (size_req 16) RM RM RM

val g_matmul_f32_tile16_ccc : fixed_repr_mmcomb_gpu_approx_ty f32 (size_req 16) CM CM CM
val g_matmul_f64_tile16_ccc : fixed_repr_mmcomb_gpu_approx_ty f64 (size_req 16) CM CM CM
val g_matmul_u32_tile16_ccc : fixed_repr_mmcomb_gpu_approx_ty u32 (size_req 16) CM CM CM
val g_matmul_u64_tile16_ccc : fixed_repr_mmcomb_gpu_approx_ty u64 (size_req 16) CM CM CM

(* GPU-side approximate GEMM - dynamically-chosen tile *)

val g_gemm_f32_rrr (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty f32 (size_req tile) RM RM RM
val g_gemm_f64_rrr (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty f64 (size_req tile) RM RM RM
val g_gemm_u32_rrr (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty u32 (size_req tile) RM RM RM
val g_gemm_u64_rrr (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty u64 (size_req tile) RM RM RM

val g_gemm_f32_ccc (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty f32 (size_req tile) CM CM CM
val g_gemm_f64_ccc (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty f64 (size_req tile) CM CM CM
val g_gemm_u32_ccc (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty u32 (size_req tile) CM CM CM
val g_gemm_u64_ccc (tile : valid_tile) : fixed_repr_gemm_gpu_approx_ty u64 (size_req tile) CM CM CM

(* GPU-side approximate GEMM - tile=32 *)

val g_gemm_f32_tile32_rrr : fixed_repr_gemm_gpu_approx_ty f32 (size_req 32) RM RM RM
val g_gemm_f64_tile32_rrr : fixed_repr_gemm_gpu_approx_ty f64 (size_req 32) RM RM RM
val g_gemm_u32_tile32_rrr : fixed_repr_gemm_gpu_approx_ty u32 (size_req 32) RM RM RM
val g_gemm_u64_tile32_rrr : fixed_repr_gemm_gpu_approx_ty u64 (size_req 32) RM RM RM

val g_gemm_f32_tile32_ccc : fixed_repr_gemm_gpu_approx_ty f32 (size_req 32) CM CM CM
val g_gemm_f64_tile32_ccc : fixed_repr_gemm_gpu_approx_ty f64 (size_req 32) CM CM CM
val g_gemm_u32_tile32_ccc : fixed_repr_gemm_gpu_approx_ty u32 (size_req 32) CM CM CM
val g_gemm_u64_tile32_ccc : fixed_repr_gemm_gpu_approx_ty u64 (size_req 32) CM CM CM

(* GPU-side approximate GEMM - tile=16 *)

val g_gemm_f32_tile16_rrr : fixed_repr_gemm_gpu_approx_ty f32 (size_req 16) RM RM RM
val g_gemm_f64_tile16_rrr : fixed_repr_gemm_gpu_approx_ty f64 (size_req 16) RM RM RM
val g_gemm_u32_tile16_rrr : fixed_repr_gemm_gpu_approx_ty u32 (size_req 16) RM RM RM
val g_gemm_u64_tile16_rrr : fixed_repr_gemm_gpu_approx_ty u64 (size_req 16) RM RM RM

val g_gemm_f32_tile16_ccc : fixed_repr_gemm_gpu_approx_ty f32 (size_req 16) CM CM CM
val g_gemm_f64_tile16_ccc : fixed_repr_gemm_gpu_approx_ty f64 (size_req 16) CM CM CM
val g_gemm_u32_tile16_ccc : fixed_repr_gemm_gpu_approx_ty u32 (size_req 16) CM CM CM
val g_gemm_u64_tile16_ccc : fixed_repr_gemm_gpu_approx_ty u64 (size_req 16) CM CM CM
