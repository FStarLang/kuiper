module Kuiper.Kernel.GEMM.TensorCore2D.To.Finish

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

inline_for_extraction noextract
fn finish
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (gC : array2 et_cd (rm m n))
  (#eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#tiles_div : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (#shared_fits : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#output_fits : squash (SZ.fits (m * n)))
  (#fragment_fits : squash (SZ.fits (tm * tn + warp_size)))
  (#tile_count_fits : squash (SZ.fits (wm * wn)))
  (#fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#scratch_fits : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (sA : array2 et_ab (rm bm bk) { core sA == fst sh })
  (sB : array2 et_ab (rm bk bn) { core sB == fst (snd sh) })
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (wid : szlt (nthr / warp_size))
  (warpRow : szlt (bm / (wm * tm)))
  (warpCol : szlt (bn / (wn * tn)))
  (#_ : squash (
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) /\
    SZ.v wid == SZ.v tid / warp_size /\
    SZ.v warpRow == SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v tn)) /\
    SZ.v warpCol == SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v tn))))
  (gwRow : enatlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow })
  (gwCol : enatlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol })
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#acc_length : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (rAcc : chest2 real (wm * tm) (wn * tn) {
    rAcc == MS.matmul
      (ematrix_subtile rA (wm * tm) k
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)) 0)
      (ematrix_subtile rB k (wn * tn) 0
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size))) })
  requires
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
      bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid **
    output_lane_live gD bm bn tm tn wm wn bid tid **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr)
  ensures
    pure (eC %~ rC) **
    gpu **
    thread_id nthr tid **
    gC |-> Frac (fC /. (m / bm * (n / bn) * nthr)) eC **
    shared_thread_live bm bn bk tm tn nthr sh tid **
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
