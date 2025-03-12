module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul
open Kuiper.MatMulCPU
open Kuiper.MatMulGPU.Type
module R = Kuiper.Matrix.Reprs

[@@CPrologue "__global__"; "KrmlPrivate"]
let k_u64_rrr
  (rows shared : szp) (cols : szp{ three_fits rows shared cols })
  : kernel_fixed_ty u64 (R.row_major rows shared) (R.row_major shared cols) (R.row_major rows cols)
  = kernel_fixed #u64 _ _ _
      #(R.crepr_row_major.map rows shared)
      #(R.crepr_row_major.map shared cols)
      #(R.crepr_row_major.map rows cols)

let matmul_u64_rrr : fixed_repr_matmul_cpu_ty u64 R.row_major R.row_major R.row_major =
  mk_fixed_repr_matmul u64 R.row_major R.row_major R.row_major
    (fun rows shared cols -> matmul_gpu_fixed (k_u64_rrr rows shared cols))

[@@CPrologue "__global__"; "KrmlPrivate"]
let k_u64_ccc
  (rows shared : szp) (cols : szp{ three_fits rows shared cols })
  : kernel_fixed_ty u64 (R.col_major rows shared) (R.col_major shared cols) (R.col_major rows cols)
  = kernel_fixed #u64 _ _ _
      #(R.crepr_col_major.map rows shared)
      #(R.crepr_col_major.map shared cols)
      #(R.crepr_col_major.map rows cols)

let matmul_u64_ccc : fixed_repr_matmul_cpu_ty u64 R.col_major R.col_major R.col_major =
  mk_fixed_repr_matmul u64 R.col_major R.col_major R.col_major
    (fun rows shared cols -> matmul_gpu_fixed (k_u64_ccc rows shared cols))
