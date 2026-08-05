module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Epilogue

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

inline_for_extraction noextract
fn epilogue_fragment_from_warp
  (#et_cd #et_acc : Type0)
  {| scd : scalar et_cd, real_like et_cd, hvc : has_vec_cpy et_cd,
     sacc : scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (#lC : RO.vlayout2 m n)
  {| str : strided_row_major lC,
     strD : strided_row_major (vtlayout_of_tlayout (rm m n)) |}
  (c : RO.roarray2 et_cd lC)
  (#_ : squash (SZ.fits (m * n)))
  (bm bn rows cols wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (#_ : squash (chunk et_cd /?+ cols))
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
  (#rC : chest2 real m n)
  (#_ : squash (eC %~ rC))
  (#lAcc : layout2 rows cols) {| T.ctlayout lAcc |}
  (acc : array2 et_acc lAcc)
  (#eAcc : chest2 et_acc rows cols)
  (#rAcc : chest2 real rows cols)
  (#_ : squash (eAcc %~ rAcc))
  (d : array2 et_cd (rm m n))
  (idx : szlt (wm * wn))
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (rows * cols + warp_size)))
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    pure (aligned 16 (RO.core c) /\ aligned 16 (T.core d) /\
          aligned_strided_row_major (chunk et_cd) str /\
          aligned_strided_row_major (chunk et_cd) strD)
  requires
    live_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      lane
  ensures
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        eAcc)
      lane **
    pure (
      epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        eAcc
      %~
      chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile rC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        rAcc)

