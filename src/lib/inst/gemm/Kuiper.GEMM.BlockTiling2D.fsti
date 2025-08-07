module Kuiper.GEMM.BlockTiling2D

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }

(* NOTE: no dynamic tile version as that would imply the use of a
   stack-allocated variable-length array, and NVCC complains
   (karamel warns too). *)

val matmul_f32_64x64x8_8x8_rrr_rr : fixed_repr_matmul_cpu_ty f32 RM RM RM
val matmul_f32_32x32x32_32x8_rrr_rr : fixed_repr_matmul_cpu_ty f32 RM RM RM
