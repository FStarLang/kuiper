module Kuiper.MatMul.Tiled.SHMem

#lang-pulse
open Kuiper
open Kuiper.Poly.MatMulCPU
open Kuiper.Poly.MatMulGPU.Type { valid_tile }
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }

(* DYNAMICALLY-CHOSEN TILE *)

val matmul_f32_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_ty f32 RM RM RM
val matmul_f64_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_ty f64 RM RM RM
val matmul_u32_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_ty u32 RM RM RM
val matmul_u64_rrr (tile : valid_tile) : fixed_repr_matmul_cpu_ty u64 RM RM RM

val matmul_f32_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_ty f32 CM CM CM
val matmul_f64_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_ty f64 CM CM CM
val matmul_u32_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_ty u32 CM CM CM
val matmul_u64_ccc (tile : valid_tile) : fixed_repr_matmul_cpu_ty u64 CM CM CM

(* SPECIALIZED TO TILE=32 *)

val matmul_f32_tile32_rrr : fixed_repr_matmul_cpu_ty f32 RM RM RM
val matmul_f64_tile32_rrr : fixed_repr_matmul_cpu_ty f64 RM RM RM
val matmul_u32_tile32_rrr : fixed_repr_matmul_cpu_ty u32 RM RM RM
val matmul_u64_tile32_rrr : fixed_repr_matmul_cpu_ty u64 RM RM RM

val matmul_f32_tile32_ccc : fixed_repr_matmul_cpu_ty f32 CM CM CM
val matmul_f64_tile32_ccc : fixed_repr_matmul_cpu_ty f64 CM CM CM
val matmul_u32_tile32_ccc : fixed_repr_matmul_cpu_ty u32 CM CM CM
val matmul_u64_tile32_ccc : fixed_repr_matmul_cpu_ty u64 CM CM CM

(* SPECIALIZED TO TILE=16 *)

val matmul_f32_tile16_rrr : fixed_repr_matmul_cpu_ty f32 RM RM RM
val matmul_f64_tile16_rrr : fixed_repr_matmul_cpu_ty f64 RM RM RM
val matmul_u32_tile16_rrr : fixed_repr_matmul_cpu_ty u32 RM RM RM
val matmul_u64_tile16_rrr : fixed_repr_matmul_cpu_ty u64 RM RM RM

val matmul_f32_tile16_ccc : fixed_repr_matmul_cpu_ty f32 CM CM CM
val matmul_f64_tile16_ccc : fixed_repr_matmul_cpu_ty f64 CM CM CM
val matmul_u32_tile16_ccc : fixed_repr_matmul_cpu_ty u32 CM CM CM
val matmul_u64_tile16_ccc : fixed_repr_matmul_cpu_ty u64 CM CM CM
