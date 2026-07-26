module Kuiper.Kernel.GEMM.TensorCore2D.To.Finish

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiper.Kernel.GEMM.TensorCore2D.To.OutputProof
open Kuiper.Kernel.GEMM.TensorCore2D.To.Cleanup

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
{
  let dims : epilogue_dims m n = {
    bm; bn; bk; tm; tn; tk; wm; wn; nthr;
    tiles_div; shared_fits; output_fits; tile_count_fits;
    scratch_fits; fragment_fits;
  };
  rewrite
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
      bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid
  as
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
      dims.bm dims.bn dims.bk dims.tm dims.tn dims.tk
      dims.wm dims.wn dims.nthr sh accFrags rAcc tid;
  rewrite
    output_lane_live gD bm bn tm tn wm wn bid tid
  as
    output_lane_live gD
      dims.bm dims.bn dims.tm dims.tn dims.wm dims.wn bid tid;
  epilogue_to #et_ab #et_cd #et_acc comb comb_r #m #n
    gC #(fC /. (m / bm * (n / bn) * nthr)) #eC #rC
    gD dims
    sh accFrags rAcc bid tid #acc_length;
  rewrite
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
      dims.bm dims.bn dims.bk dims.tm dims.tn dims.tk
      dims.wm dims.wn dims.nthr sh accFrags rAcc tid
  as
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
      bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;
  rewrite
    output_lane_approximates gD
      dims.bm dims.bn dims.tm dims.tn dims.wm dims.wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC dims.bm dims.bn
            (bid / (n / dims.bn)) (bid % (n / dims.bn)))
          (dims.wm * dims.tm) (dims.wn * dims.tn)
          ((tid / warp_size) / (dims.bn / (dims.wn * dims.tn)))
          ((tid / warp_size) % (dims.bn / (dims.wn * dims.tn))))
        rAcc)
  as
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn
            (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
        rAcc);
  unfold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;
  normalize_output comb_r gD
    bm bn bk tm tn tk wm wn rA rB rC nthr
    bid tid mrow mcol wid warpRow warpCol gwRow gwCol rAcc;
  cleanup #et_ab #et_acc
    bm bn bk tm tn tk wm wn nthr sh sA sB tid accFrags rAcc;
}
