module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_u32 = kernel #u32 R.c_row_major R.c_row_major R.c_row_major

let matmul_u32 : matmul_ty u32 R.c_row_major R.c_row_major R.c_row_major =
  matmul R.c_row_major R.c_row_major R.c_row_major kernel_u32 
