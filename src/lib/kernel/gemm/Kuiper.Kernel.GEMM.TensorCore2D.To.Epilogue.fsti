module Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

inline_for_extraction noextract
let sz_succ (x : SZ.t { SZ.fits (x + 1) }) : SZ.t = x +^ 1sz

let fragarrayAcc_approximates
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm wn : nat)
  ([@@@mkey] arr : array (fragment et FragAcc tm tn tk FragLAcc) {
    Pulse.Lib.Array.length arr == wm * wn})
  (rm : chest2 real (wm * tm) (wn * tn))
  : slprop
= exists* (em : seq (chest2 et tm tn)).
    arr |-> em **
    pure (
      Seq.length em == wm * wn /\
      forall (i : natlt wm) (j : natlt wn).
        Seq.index em (i * wn + j) %~
          ematrix_subtile rm tm tn i j)

inline_for_extraction noextract
fn epilogue_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (gC : array2 et_cd (rm m n))
  (#fC : perm)
  (#eC : chest2 et_cd m n)
  (#rC : chest2 real m n)
  (#_ : squash (eC %~ rC))
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn nthr : szp{
    constraints bm bn bk tm tn tk wm wn /\
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (SZ.fits (tm * tn + warp_size)))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  preserves
    gpu **
    thread_id nthr tid **
    gC |-> Frac fC eC **
    fragarrayAcc_approximates wm wn accFrags rAcc
  requires
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    output_lane_live gD bm bn tm tn wm wn bid tid
  ensures
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn
            (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
        rAcc)
