module Kuiper.MatMul.F64

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_f64 = kernel #f64 R.row_major R.row_major R.row_major

let matmul_f64 : matmul_ty f64 R.row_major R.row_major R.row_major =
  matmul kernel_f64
