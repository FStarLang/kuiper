module Kuiper.Kernel.GEMM.TensorCore2D.To.KLoopState

open Kuiper

module SZ = Kuiper.SizeT

open Kuiper.EMatrix
open Kuiper.Spec.GEMM

let barrier_iteration (v : SZ.t) : nat = 2 * SZ.v v

let tile_barrier_iteration (k bk : szp) : nat =
  2 * (SZ.v k / SZ.v bk)

let tiled_partial_matmul
  (#rows #shared #cols #tm #tk #tn : nat)
  (z : chest2 real tm tn)
  (a : chest2 (chest2 real tm tk) rows shared)
  (b : chest2 (chest2 real tk tn) shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (upto : nat { upto <= shared })
  : chest2 real tm tn
= __gmatmul_single z matmul matplus a b row col upto
