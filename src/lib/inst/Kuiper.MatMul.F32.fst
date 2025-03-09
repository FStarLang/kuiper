module Kuiper.MatMul.F32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"; "KrmlPrivate"]
let kernel_f32
  : kernel_ty _ _ _ _
  = kernel #f32 R.row_major R.row_major R.row_major

let matmul_f32 : matmul_ty f32 R.row_major R.row_major R.row_major =
  matmul kernel_f32
