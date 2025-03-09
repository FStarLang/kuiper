module Kuiper.MatMul.F32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_f32 = kernel #f32 R.c_row_major R.c_row_major R.c_row_major

let matmul_f32 : matmul_ty f32 R.c_row_major R.c_row_major R.c_row_major =
  matmul kernel_f32
