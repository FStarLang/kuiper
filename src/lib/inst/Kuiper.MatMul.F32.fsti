module Kuiper.MatMul.F32

#lang-pulse
open Kuiper
open Kuiper.MatMul
module R = Kuiper.Matrix.Reprs

val matmul_f32 : matmul_ty f32 R.row_major R.row_major R.row_major
