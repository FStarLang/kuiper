module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_u64 = kernel #u64 R.c_row_major R.c_row_major R.c_row_major

let matmul_u64 : matmul_ty u64 R.c_row_major R.c_row_major R.c_row_major =
  matmul kernel_u64
