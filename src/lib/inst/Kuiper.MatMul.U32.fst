module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32
  : kernel_ty _ _ _ _
  = kernel #u32 R.row_major R.row_major R.row_major

let matmul_u32 : matmul_ty u32 R.row_major R.row_major R.row_major =
  matmul kernel_u32

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32'
  : kernel_ty _ _ _ _
  = kernel #u32 R.col_major R.col_major R.col_major

let matmul_u32' : matmul_ty u32 R.col_major R.col_major R.col_major =
  matmul kernel_u32'

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32''
  : kernel_ty _ _ _ _
  = kernel #u32 R.col_major R.col_major R.row_major

let matmul_u32'' : matmul_ty u32 R.col_major R.col_major R.row_major =
  matmul kernel_u32''

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_u32_mirror
  : kernel_ty _ _ _ _
  = kernel #u32 R.row_major_mirror R.row_major_mirror R.row_major_mirror

let matmul_u32_mirror : matmul_ty u32 R.row_major_mirror R.row_major_mirror R.row_major_mirror =
  matmul kernel_u32_mirror
