module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMulCPU
module R = Kuiper.Matrix.Reprs

val matmul_u32_rrr : fixed_repr_matmul_cpu_ty u32 R.row_major R.row_major R.row_major
