module Kuiper.MatMul.F64

#lang-pulse
open Kuiper
open Kuiper.MatMul
module R = Kuiper.Matrix.Reprs

val matmul_f64 : matmul_ty f64 R.row_major R.row_major R.row_major
