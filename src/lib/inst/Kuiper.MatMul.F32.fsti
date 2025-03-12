module Kuiper.MatMul.F32

#lang-pulse
open Kuiper
open Kuiper.MatMulCPU
module R = Kuiper.Matrix.Reprs

val matmul_f32_rrr : fixed_repr_matmul_cpu_ty f32 R.row_major R.row_major R.row_major
