module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u64
  : kernel_ty _ _ _ _
  = kernel #u64 R.row_major R.row_major R.row_major

let matmul_u64 : matmul_ty u64 R.row_major R.row_major R.row_major =
  matmul kernel_u64
