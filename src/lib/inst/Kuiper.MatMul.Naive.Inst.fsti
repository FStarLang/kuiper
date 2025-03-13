module Kuiper.MatMul.Naive.Inst

#lang-pulse
open Kuiper
open Kuiper.MatMulCPU
module R = Kuiper.Matrix.Reprs

val matmul_f32_rrr : fixed_repr_matmul_cpu_ty f32 R.row_major R.row_major R.row_major
val matmul_f64_rrr : fixed_repr_matmul_cpu_ty f64 R.row_major R.row_major R.row_major
val matmul_u32_rrr : fixed_repr_matmul_cpu_ty u32 R.row_major R.row_major R.row_major
val matmul_u64_rrr : fixed_repr_matmul_cpu_ty u64 R.row_major R.row_major R.row_major

val matmul_f32_ccc : fixed_repr_matmul_cpu_ty f32 R.col_major R.col_major R.col_major
val matmul_f64_ccc : fixed_repr_matmul_cpu_ty f64 R.col_major R.col_major R.col_major
val matmul_u32_ccc : fixed_repr_matmul_cpu_ty u32 R.col_major R.col_major R.col_major
val matmul_u64_ccc : fixed_repr_matmul_cpu_ty u64 R.col_major R.col_major R.col_major
