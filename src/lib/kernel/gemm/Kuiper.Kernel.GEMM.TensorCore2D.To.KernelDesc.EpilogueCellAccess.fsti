module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellAccess

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell

noextract
ghost fn output_cell_to_global
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (mi : natlt wm)
  (nj : natlt wn)
  (i : natlt tm)
  (j : natlt tn)
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : natlt n {
    globalCol == tiled_cell n bn (bid % (n / bn))
      (tiled_cell bn (wn * tn) (wid % (bn / (wn * tn)))
        (tiled_cell (wn * tn) tn nj j)) })
  (f : perm)
  (v : et)
  requires
    tensor_pts_to_cell
      (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
      #f (idx2 i j) v
  ensures
    tensor_pts_to_cell gD #f (idx2 globalRow globalCol) v

noextract
ghost fn global_cell_to_output
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (mi : natlt wm)
  (nj : natlt wn)
  (i : natlt tm)
  (j : natlt tn)
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : natlt n {
    globalCol == tiled_cell n bn (bid % (n / bn))
      (tiled_cell bn (wn * tn) (wid % (bn / (wn * tn)))
        (tiled_cell (wn * tn) tn nj j)) })
  (f : perm)
  (v : et)
  requires
    tensor_pts_to_cell gD #f (idx2 globalRow globalCol) v
  ensures
    tensor_pts_to_cell
      (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
      #f (idx2 i j) v
