module Kuiper.MatMul.F64

#lang-pulse
open Kuiper
open Kuiper.MatMulCPU
module R = Kuiper.Matrix.Reprs

val matmul_f64_rrr : fixed_repr_matmul_cpu_ty f64 R.row_major R.row_major R.row_major
