module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

[@@erasable]
val epilogue_fragment_target
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : nat)
  (eC : chest2 et_cd m n)
  (bm bn rows cols wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * rows)))
  (warpCol : natlt (bn / (wn * cols)))
  (idx : natlt (wm * wn))
  (eAcc : chest2 et_acc rows cols)
  : chest2 et_cd rows cols

val epilogue_fragment_target_eq
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : nat)
  (eC : chest2 et_cd m n)
  (bm bn rows cols wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * rows)))
  (warpCol : natlt (bn / (wn * cols)))
  (idx : natlt (wm * wn))
  (eAcc : chest2 et_acc rows cols)
  : Lemma (
      epilogue_fragment_target comb eC
        bm bn rows cols wm wn
        mrow mcol warpRow warpCol idx eAcc
      ==
      epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn mrow mcol)
            (wm * rows) (wn * cols) warpRow warpCol)
          rows cols (idx / wn) (idx % wn))
        eAcc)

inline_for_extraction noextract
fn epilogue_cell_update
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : szp)
  (c : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn rows cols wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (warpRow : szlt (bm / (wm * rows)))
  (warpCol : szlt (bn / (wn * cols)))
  (bid : szlt (m / bm * (n / bn)))
  (wid : szlt (bm / (wm * rows) * (bn / (wn * cols))))
  (#_ : squash (
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) /\
    SZ.v warpRow == SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v cols)) /\
    SZ.v warpCol == SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v cols))))
  (#fC : perm)
  (#eC : chest2 et_cd m n)
  (#lAcc : layout2 rows cols) {| T.ctlayout lAcc |}
  (acc : array2 et_acc lAcc)
  (#eAcc : chest2 et_acc rows cols)
  (d : array2 et_cd (rm m n))
  (idx : szlt (wm * wn))
  (row : szlt rows)
  (col : szlt cols)
  (#old : erased et_cd)
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (reveal old)
  ensures
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (acc2
        (epilogue_fragment_target comb eC
          bm bn rows cols wm wn
          (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
          (SZ.v idx) eAcc)
        (SZ.v row) (SZ.v col))
