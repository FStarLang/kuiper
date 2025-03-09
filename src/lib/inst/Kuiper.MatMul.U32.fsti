module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMul
module R = Kuiper.Matrix.Reprs

val matmul_u32 : matmul_ty u32 R.c_row_major R.c_row_major R.c_row_major
