module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul
module R = Kuiper.Matrix.Reprs

val matmul_u64_rrr : matmul_ty u64 R.row_major R.row_major R.row_major
val matmul_u64_ccc : matmul_ty u64 R.col_major R.col_major R.col_major
val matmul_u64_ccr : matmul_ty u64 R.col_major R.col_major R.row_major
val matmul_u64_rrc : matmul_ty u64 R.row_major R.row_major R.col_major
