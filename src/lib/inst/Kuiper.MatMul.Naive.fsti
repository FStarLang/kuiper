module Kuiper.MatMul.Naive

#lang-pulse
open Kuiper
open Kuiper.Poly.MatMulCPU
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }

val matmul_f32_rrr : fixed_repr_matmul_cpu_ty f32 RM RM RM
val matmul_f64_rrr : fixed_repr_matmul_cpu_ty f64 RM RM RM
val matmul_u32_rrr : fixed_repr_matmul_cpu_ty u32 RM RM RM
val matmul_u64_rrr : fixed_repr_matmul_cpu_ty u64 RM RM RM

val matmul_f32_ccc : fixed_repr_matmul_cpu_ty f32 CM CM CM
val matmul_f64_ccc : fixed_repr_matmul_cpu_ty f64 CM CM CM
val matmul_u32_ccc : fixed_repr_matmul_cpu_ty u32 CM CM CM
val matmul_u64_ccc : fixed_repr_matmul_cpu_ty u64 CM CM CM

val g_matmul_f32_rrr : fixed_repr_matmul_gpu_ty f32 RM RM RM
val g_matmul_f64_rrr : fixed_repr_matmul_gpu_ty f64 RM RM RM
val g_matmul_u32_rrr : fixed_repr_matmul_gpu_ty u32 RM RM RM
val g_matmul_u64_rrr : fixed_repr_matmul_gpu_ty u64 RM RM RM

val g_matmul_f32_ccc : fixed_repr_matmul_gpu_ty f32 CM CM CM
val g_matmul_f64_ccc : fixed_repr_matmul_gpu_ty f64 CM CM CM
val g_matmul_u32_ccc : fixed_repr_matmul_gpu_ty u32 CM CM CM
val g_matmul_u64_ccc : fixed_repr_matmul_gpu_ty u64 CM CM CM
