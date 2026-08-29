module Kuiper.Kernel.GEMM.TensorCore2D.To.Finish

#lang-pulse

open Kuiper

open Kuiper.Array.Vectorized { has_vec_cpy }
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

#push-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

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
{
  unfold shared_thread_live bm bn bk tm tn nthr sh tid;
  let (sarA, (sarB, (sarAcc, sarTail))) = sh;
  unfold live_c_shmem sarA #(1.0R /. nthr);
  unfold live_c_shmem sarB #(1.0R /. nthr);
  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  tensor_abs' (rm bm bk) sarA;
  let sA = from_array (rm bm bk) sarA;
  rewrite each _ as sA;
  tensor_abs' (rm bk bn) sarB;
  let sB = from_array (rm bk bn) sarB;
  rewrite each _ as sB;

  rewrite
    (exists* (x : chest2 _ _ _). sA |-> Frac (1.0R /. nthr) x) **
    (exists* (x : chest2 _ _ _). sB |-> Frac (1.0R /. nthr) x)
  as
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr);
  rewrite
    scratch_tile_live bm bn bk tm tn nthr
      (sarA, (sarB, (sarAcc, sarTail))) tid
  as
    scratch_tile_live bm bn bk tm tn nthr sh tid;
  (sA, sB)
}

#pop-options

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
{
  fold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC fC eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;
}

#push-options "--fuel 1 --ifuel 1 --z3rlimit 100"
let tile_offset_bound
  (whole : pos)
  (tile : pos {tile /? whole})
  (t : natlt (whole / tile))
  (i : natlt tile)
  : Lemma (t * tile + i < whole)
  = Kuiper.Divides.lemma_divides_exact tile whole;
    FStar.Matrix.flattened_index_is_under_flattened_size
      (whole / tile) tile t i

let nested_comb_tile_eq
  (#m #n : pos)
  (comb_r : binop real)
  (rC rM : chest2 real m n)
  (bm : pos {bm /? m})
  (bn : pos {bn /? n})
  (wm_tm : pos {wm_tm /? bm})
  (wn_tn : pos {wn_tn /? bn})
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / wm_tm))
  (warpCol : natlt (bn / wn_tn))
  (gwRow : enatlt (m / wm_tm) {
    gwRow * wm_tm == mrow * bm + warpRow * wm_tm })
  (gwCol : enatlt (n / wn_tn) {
    gwCol * wn_tn == mcol * bn + warpCol * wn_tn })
  (rMTile : chest2 real wm_tm wn_tn {
    rMTile == ematrix_subtile rM wm_tm wn_tn gwRow gwCol })
  : Lemma (
      chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn mrow mcol)
          wm_tm wn_tn warpRow warpCol)
        rMTile
      ==
      ematrix_subtile
        (ematrix_subtile (chest_comb comb_r rC rM)
          bm bn mrow mcol)
        wm_tm wn_tn warpRow warpCol)
  = let lhs =
      chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn mrow mcol)
          wm_tm wn_tn warpRow warpCol)
        rMTile in
    let rhs =
      ematrix_subtile
        (ematrix_subtile (chest_comb comb_r rC rM)
          bm bn mrow mcol)
        wm_tm wn_tn warpRow warpCol in
    let aux (i : natlt wm_tm) (j : natlt wn_tn)
      : Lemma (acc2 lhs i j == acc2 rhs i j)
      = assert (gwRow * wm_tm + i ==
                  mrow * bm + warpRow * wm_tm + i);
        assert (gwCol * wn_tn + j ==
                  mcol * bn + warpCol * wn_tn + j);
        tile_offset_bound bm wm_tm warpRow i;
        tile_offset_bound bn wn_tn warpCol j;
        tile_offset_bound m bm mrow (warpRow * wm_tm + i);
        tile_offset_bound n bn mcol (warpCol * wn_tn + j);
        subtile_acc2
          (ematrix_subtile (chest_comb comb_r rC rM)
            bm bn mrow mcol)
          wm_tm wn_tn warpRow warpCol i j;
        subtile_acc2 (chest_comb comb_r rC rM)
          bm bn mrow mcol
          (warpRow * wm_tm + i) (warpCol * wn_tn + j);
        chest_comb_acc2 comb_r rC rM
          (mrow * bm + warpRow * wm_tm + i)
          (mcol * bn + warpCol * wn_tn + j);
        assert (
          acc2 rhs i j ==
          comb_r
            (acc2 rC
              (mrow * bm + warpRow * wm_tm + i)
              (mcol * bn + warpCol * wn_tn + j))
            (acc2 rM
              (mrow * bm + warpRow * wm_tm + i)
              (mcol * bn + warpCol * wn_tn + j)));
        calc (==) {
          acc2 lhs i j;
          == {}
          comb_r
            (acc2 rC
              (mrow * bm + warpRow * wm_tm + i)
              (mcol * bn + warpCol * wn_tn + j))
            (acc2 rMTile i j);
          == {}
          comb_r
            (acc2 rC (gwRow * wm_tm + i) (gwCol * wn_tn + j))
            (acc2 rM (gwRow * wm_tm + i) (gwCol * wn_tn + j));
          == {}
          acc2 rhs i j;
        } in
    Classical.forall_intro_2 aux;
    Kuiper.EMatrix.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs
#pop-options

#push-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#push-options "--z3rlimit 200"

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
      (epilogue_warp_output comb_r rC
        bm bn tm tn wm wn bid tid rAcc)
  ensures
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
{
  MS.matmul_decompose_lemma rA rB
    (wm * tm) (wn * tn) gwRow gwCol;
  let rOutLocal =
    chest_comb comb_r
      (ematrix_subtile
        (ematrix_subtile rC bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
      rAcc;
  let rOutTarget =
    ematrix_subtile
      (ematrix_subtile
        (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
  (* The precondition refers to the warp's output tile through the
     [epilogue_warp_output] abbreviation.  Peel it off here, once, into
     the spelled-out local [rOutLocal] used by the rest of the proof.
     Keeping the abbreviation folded in the *specification* is what
     keeps the call to this function (and to [epilogue_to]) cheap for
     F*'s core typechecker: see the comment on [epilogue_warp_output]. *)
  assert pure (
    epilogue_warp_output comb_r rC
      bm bn tm tn wm wn bid tid rAcc == rOutLocal);
  rewrite each
    (epilogue_warp_output comb_r rC
      bm bn tm tn wm wn bid tid rAcc)
  as rOutLocal;
  assert pure (
    (SZ.v bm / (SZ.v wm * SZ.v tm)) *
      (SZ.v wm * SZ.v tm) == SZ.v bm);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v tn)) *
      (SZ.v wn * SZ.v tn) == SZ.v bn);
  // Spell out the two nonlinear rearrangements: with one query per proof
  // obligation these are no longer found by E-matching alone.
  FStar.Math.Lemmas.distributivity_add_left
    (SZ.v mrow * (SZ.v bm / (SZ.v wm * SZ.v tm))) (SZ.v warpRow)
    (SZ.v wm * SZ.v tm);
  FStar.Math.Lemmas.paren_mul_right
    (SZ.v mrow) (SZ.v bm / (SZ.v wm * SZ.v tm)) (SZ.v wm * SZ.v tm);
  assert pure (
    gwRow * (SZ.v wm * SZ.v tm) ==
    SZ.v mrow * SZ.v bm +
      SZ.v warpRow * (SZ.v wm * SZ.v tm));
  FStar.Math.Lemmas.distributivity_add_left
    (SZ.v mcol * (SZ.v bn / (SZ.v wn * SZ.v tn))) (SZ.v warpCol)
    (SZ.v wn * SZ.v tn);
  FStar.Math.Lemmas.paren_mul_right
    (SZ.v mcol) (SZ.v bn / (SZ.v wn * SZ.v tn)) (SZ.v wn * SZ.v tn);
  assert pure (
    gwCol * (SZ.v wn * SZ.v tn) ==
    SZ.v mcol * SZ.v bn +
      SZ.v warpCol * (SZ.v wn * SZ.v tn));
  assert pure (
    warp_tile_i #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size) == gwRow);
  assert pure (
    warp_tile_j #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size) == gwCol);
  let rMatmul =
    ematrix_subtile (MS.matmul rA rB)
      (wm * tm) (wn * tn) gwRow gwCol;
  assert pure (rAcc == rMatmul);
  nested_comb_tile_eq comb_r rC (MS.matmul rA rB)
    bm bn (wm * tm) (wn * tn)
    mrow mcol warpRow warpCol gwRow gwCol rMatmul;
  assert pure (rOutLocal == rOutTarget);
  rewrite
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid rOutLocal
  as
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid rOutTarget;
  rewrite each rOutTarget as
    ematrix_subtile
      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
}

#pop-options
#pop-options

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
{
  with em1. unfold FB.bp_sharing sA em1 nthr;
  with em2. unfold FB.bp_sharing sB em2 nthr;
  unfold fragarrayAcc_approximates wm wn accFrags rAcc;
  with vaccumFrags. assert accFrags |-> vaccumFrags;
  drop_ (accFrags |-> vaccumFrags);
  tensor_concr sA;
  rewrite each core sA as fst sh;
  tensor_concr sB;
  rewrite each core sB as fst (snd sh);
  fold live_c_shmem (fst sh) #(1.0R /. nthr);
  fold live_c_shmem (fst (snd sh)) #(1.0R /. nthr);
  fold shared_thread_live bm bn bk tm tn nthr sh tid;
}

#push-options "--fuel 1 --ifuel 1"
#push-options "--z3rlimit 15"

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
{
  epilogue_to #et_ab #et_cd #et_acc comb comb_r #m #n
    gC #(fC /. (m / bm * (n / bn) * nthr)) #eC #rC
    gD {
      bm; bn; bk; tm; tn; tk; wm; wn; nthr;
      tiles_div; shared_fits; output_fits; tile_count_fits;
      scratch_fits; fragment_fits;
    }
    sh accFrags rAcc bid tid #acc_length;
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

#pop-options
#pop-options
