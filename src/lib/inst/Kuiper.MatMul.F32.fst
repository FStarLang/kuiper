module Kuiper.MatMul.F32

#lang-pulse
open Kuiper
open Kuiper.MatMul
open Kuiper.MatMulCPU
open Kuiper.MatMulGPU.Type
module R = Kuiper.Matrix.Reprs

[@@CPrologue "__global__"; "KrmlPrivate"]
let k_f32_rrr
  (rows shared : szp) (cols : szp{ three_fits rows shared cols })
  : kernel_fixed_ty f32 (R.row_major rows shared) (R.row_major shared cols) (R.row_major rows cols)
  = kernel_fixed #f32 _ _ _
      #(R.crepr_row_major.map rows shared)
      #(R.crepr_row_major.map shared cols)
      #(R.crepr_row_major.map rows cols)

let matmul_f32_rrr : fixed_repr_matmul_cpu_ty f32 R.row_major R.row_major R.row_major =
  mk_fixed_repr_matmul f32 R.row_major R.row_major R.row_major
    (fun rows shared cols -> matmul_gpu_fixed (k_f32_rrr rows shared cols))
