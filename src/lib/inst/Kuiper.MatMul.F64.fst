module Kuiper.MatMul.F64

#lang-pulse
open Kuiper
open Kuiper.MatMul
open Kuiper.MatMulCPU
open Kuiper.MatMulGPU.Type
module R = Kuiper.Matrix.Reprs

[@@CPrologue "__global__"; "KrmlPrivate"]
let k_f64_rrr
  (rows shared : szp) (cols : szp{ three_fits rows shared cols })
  : kernel_fixed_ty f64 (R.row_major rows shared) (R.row_major shared cols) (R.row_major rows cols)
  = kernel_fixed #f64 _ _ _ 
      #(R.crepr_row_major.map rows shared)
      #(R.crepr_row_major.map shared cols)
      #(R.crepr_row_major.map rows cols)

let matmul_f64_rrr : fixed_repr_matmul_cpu_ty f64 R.row_major R.row_major R.row_major =
  mk_fixed_repr_matmul f64 R.row_major R.row_major R.row_major
    (fun rows shared cols -> matmul_gpu_fixed (k_f64_rrr rows shared cols))
