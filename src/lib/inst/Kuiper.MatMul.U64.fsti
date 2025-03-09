module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul
module R = Kuiper.Matrix.Reprs

val matmul_u64 : matmul_ty u64 R.row_major R.row_major R.row_major
