module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u64_rrr
  : kernel_ty _ _ _ _
  = kernel #u64 R.row_major R.row_major R.row_major

let matmul_u64_rrr : matmul_ty u64 R.row_major R.row_major R.row_major =
  matmul kernel_u64_rrr

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u64_ccc
  : kernel_ty _ _ _ _
  = kernel #u64 R.col_major R.col_major R.col_major

let matmul_u64_ccc : matmul_ty u64 R.col_major R.col_major R.col_major =
  matmul kernel_u64_ccc

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u64_ccr
  : kernel_ty _ _ _ _
  = kernel #u64 R.col_major R.col_major R.row_major

let matmul_u64_ccr : matmul_ty u64 R.col_major R.col_major R.row_major =
  matmul kernel_u64_ccr

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u64_rrc
  : kernel_ty _ _ _ _
  = kernel #u64 R.row_major R.row_major R.col_major

let matmul_u64_rrc : matmul_ty u64 R.row_major R.row_major R.col_major =
  matmul kernel_u64_rrc
