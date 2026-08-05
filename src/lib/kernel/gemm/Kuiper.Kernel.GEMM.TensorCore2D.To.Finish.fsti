module Kuiper.Kernel.GEMM.TensorCore2D.To.Finish

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy }
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
fn setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#shared_fits : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#scratch_fits : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#warp_div : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (tid : szlt nthr)
  requires
    gpu **
    shared_thread_live bm bn bk tm tn nthr sh tid
  returns s :
    (array2 et_ab (rm bm bk) & array2 et_ab (rm bk bn))
  ensures
    pure (core s._1 == fst sh /\ core s._2 == fst (snd sh)) **
    gpu **
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    (exists* em1. FB.bp_sharing s._1 em1 nthr) **
    (exists* em2. FB.bp_sharing s._2 em2 nthr)

noextract
ghost fn prepare_epilogue
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (#m #n : szp)
  (gC : array2 et_cd (rm m n))
  (fC : perm)
  (eC : chest2 et_cd m n)
  (rC : chest2 real m n)
  (bm bn bk tm tn tk wm wn nthr : szp {
    constraints bm bn bk tm tn tk wm wn /\
    Kuiper.SizeT.v nthr ==
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (Kuiper.SizeT.fits (bm * bk) /\
                Kuiper.SizeT.fits (bk * bn)))
  (#_ : squash (Kuiper.SizeT.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc) {
    Pulse.Lib.Array.length accFrags == wm * wn })
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (tid : szlt nthr)
  requires
    pure (eC %~ rC) **
    gpu **
    thread_id nthr tid **
    gC |-> Frac fC eC **
    fragarrayAcc_approximates wm wn accFrags rAcc **
    scratch_tile_live bm bn bk tm tn nthr sh tid
  ensures
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid

noextract
ghost fn normalize_output
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
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
  (rAcc : chest2 real (wm * tm) (wn * tn) {
    rAcc == MS.matmul
      (ematrix_subtile rA (wm * tm) k
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)) 0)
      (ematrix_subtile rB k (wn * tn) 0
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size))) })
  requires
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn
            (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
        rAcc)
  ensures
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)

noextract
ghost fn cleanup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_acc, real_like et_acc |}
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (nthr : szp {
    Kuiper.SizeT.v nthr ==
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (Kuiper.SizeT.fits (bm * bk) /\
                Kuiper.SizeT.fits (bk * bn)))
  (#_ : squash (Kuiper.SizeT.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (sA : array2 et_ab (rm bm bk) { core sA == fst sh })
  (sB : array2 et_ab (rm bk bn) { core sB == fst (snd sh) })
  (tid : szlt nthr)
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (rAcc : chest2 real (wm * tm) (wn * tn))
  requires
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    fragarrayAcc_approximates wm wn accFrags rAcc **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr)
  ensures
    shared_thread_live bm bn bk tm tn nthr sh tid

#push-options "--z3rlimit 10"
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
  (#warp_div : squash (warp_size /?+ nthr))
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
#pop-options
