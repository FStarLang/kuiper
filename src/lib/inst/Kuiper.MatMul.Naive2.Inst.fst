module Kuiper.MatMul.Naive2.Inst

#lang-pulse
open Kuiper
open Kuiper.MatMul.Naive2
open Kuiper.MatMulCPU
open Kuiper.MatMulGPU.Type
module R = Kuiper.Matrix.Reprs

let matmul_f32_rrr : fixed_repr_matmul_cpu_ty f32 R.row_major R.row_major R.row_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_f64_rrr : fixed_repr_matmul_cpu_ty f64 R.row_major R.row_major R.row_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_u32_rrr : fixed_repr_matmul_cpu_ty u32 R.row_major R.row_major R.row_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_u64_rrr : fixed_repr_matmul_cpu_ty u64 R.row_major R.row_major R.row_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_f32_ccc : fixed_repr_matmul_cpu_ty f32 R.col_major R.col_major R.col_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_f64_ccc : fixed_repr_matmul_cpu_ty f64 R.col_major R.col_major R.col_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_u32_ccc : fixed_repr_matmul_cpu_ty u32 R.col_major R.col_major R.col_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _

let matmul_u64_ccc : fixed_repr_matmul_cpu_ty u64 R.col_major R.col_major R.col_major =
  specialize_to_type_and_reprs matmul_gpu _ _ _ _
