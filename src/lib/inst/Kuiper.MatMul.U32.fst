module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32_rrr
  : kernel_ty _ _ _ _
  = kernel #u32 R.row_major R.row_major R.row_major

let matmul_u32_rrr : matmul_ty u32 R.row_major R.row_major R.row_major =
  matmul kernel_u32_rrr

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32_ccc
  : kernel_ty _ _ _ _
  = kernel #u32 R.col_major R.col_major R.col_major

let matmul_u32_ccc : matmul_ty u32 R.col_major R.col_major R.col_major =
  matmul kernel_u32_ccc

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32_ccr
  : kernel_ty _ _ _ _
  = kernel #u32 R.col_major R.col_major R.row_major

let matmul_u32_ccr : matmul_ty u32 R.col_major R.col_major R.row_major =
  matmul kernel_u32_ccr
