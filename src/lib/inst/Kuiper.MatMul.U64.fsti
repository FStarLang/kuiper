module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMulCPU
module R = Kuiper.Matrix.Reprs

val matmul_u64_rrr : fixed_repr_matmul_cpu_ty u64 R.row_major R.row_major R.row_major
