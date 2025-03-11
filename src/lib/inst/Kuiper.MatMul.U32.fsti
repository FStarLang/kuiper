module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMul
module R = Kuiper.Matrix.Reprs

val matmul_u32_rrr : matmul_ty u32 R.row_major R.row_major R.row_major
val matmul_u32_ccc : matmul_ty u32 R.col_major R.col_major R.col_major
val matmul_u32_ccr : matmul_ty u32 R.col_major R.col_major R.row_major
