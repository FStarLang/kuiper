module Kuiper.Kernel.GEMM.FlipFlopBarrier2

(* This module defines a barrier contract used by GEMMs that operate
   on Array2 (Tensor-backed) matrices. *)

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Tensor.Tiling

open Kuiper.Tensor
module B = Kuiper.Barrier
module SZ = Kuiper.SizeT
module CV = Kuiper.Kernel.GEMM.Copy.Vec2

let own_strided_chunks
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (em : chest2 et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
    tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)

let live_strided_chunks
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  exists* em.
    own_strided_chunks m em nthr tid

let bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (em : chest2 et rows cols)
  (nthr : pos)
  : slprop
  = m |-> Frac (1.0R /. nthr) em

(* The barrier guard gives [it < 2 * shared / bk]; the subtile index [it / 2]
   must be below [shared / bk].  With per-goal SMT queries the solver no longer
   gets these nonlinear division facts for free, so we spell them out. *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
(* Stated with a trigger: [barrier_p]/[barrier_q] get unfolded (and hence
   re-typechecked) at their use sites, where an explicit lemma call inside the
   definition body is no longer available. *)
let half_lt_quot (it shared bk : nat)
  : Lemma (requires shared > 0 /\ bk > 0 /\ shared % bk = 0 /\ it < 2 * shared / bk)
          (ensures it / 2 < shared / bk)
          [SMTPat (it / 2); SMTPat (shared / bk)]
  = let q = shared / bk in
    FStar.Math.Lemmas.lemma_div_exact shared bk;          // shared == bk * q
    assert (2 * shared == (2 * q) * bk);
    FStar.Math.Lemmas.multiple_division_lemma (2 * q) bk; // ((2*q)*bk)/bk == 2*q
    FStar.Math.Lemmas.euclidean_division_definition it 2  // it == 2*(it/2) + it%2

(* Splitting the flat block id by the column-block count gives a valid row
   index.  Keep the quotient and product as triggers because [barrier_p] and
   [barrier_q] are unfolded in downstream barrier proofs. *)
let div_lt_mul_pat (a p : nat) (q : pos)
  : Lemma (requires a < p * q)
          (ensures 0 <= a / q /\ a / q < p)
          [SMTPat (a / q); SMTPat (p * q)]
  = FStar.Math.Lemmas.nat_over_pos_is_nat a q;
    if a / q >= p then begin
      FStar.Math.Lemmas.lemma_mult_le_right q p (a / q);
      FStar.Math.Lemmas.euclidean_division_definition a q
    end
#pop-options

let barrier_p
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    if it >= 2 * shared / bk then
      emp
    else if even it then
      (exists* em1. bp_sharing m1 em1 nthr) **
      (exists* em2. bp_sharing m2 em2 nthr)
    else
      let mrow = bid / (cols/bn) in
      let mcol = bid % (cols/bn) in
      div_lt_mul_pat bid (rows/bm) (cols/bn);
      half_lt_quot it shared bk;
      own_strided_chunks m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr tid **
      own_strided_chunks m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr tid

let barrier_q
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    if it >= 2 * shared / bk then
      emp
    else if even it then
      live_strided_chunks m1 nthr tid **
      live_strided_chunks m2 nthr tid
    else
      let mrow = bid / (cols/bn) in
      let mcol = bid % (cols/bn) in
      div_lt_mul_pat bid (rows/bm) (cols/bn);
      half_lt_quot it shared bk;
      bp_sharing m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr **
      bp_sharing m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr

let contract
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk))
  (sar2 : larray etB (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.contract nthr =
{
  B.rin  = barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
  B.rout = barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
}

let barrier_tok
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk))
  (sar2 : larray etB (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : slprop
  = B.barrier_tok (contract eA eB l1 l2 sar1 sar2 nthr bid)

(* The proof of correctness. *)
ghost
fn barrier_p_to_q_transform
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk))
  (sar2 : larray etB (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (#_ : squash (chunk etB /?+ bn))
  (#_ : squash (chunk etA /?+ bk))
  (#_ : squash (chunk etA * nthr /?+ (bm * bk)))
  (#_ : squash (chunk etB * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
  ensures
    forall+ (tid : natlt nthr).
      barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid

(* Per-thread helpers for odd iterations. *)
ghost
fn fold_barrier_p_odd
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (mrow : nat{mrow == bid / (cols/bn)})
  (mcol : nat{mcol == bid % (cols/bn)})
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    own_strided_chunks m1 (ematrix_subtile eA bm bk mrow bkIdx) nthr tid **
    own_strided_chunks m2 (ematrix_subtile eB bk bn bkIdx mcol) nthr tid
  ensures
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx + 1) tid

ghost
fn unfold_barrier_q_odd
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (mrow : nat{mrow == bid / (cols/bn)})
  (mcol : nat{mcol == bid % (cols/bn)})
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx + 1) tid
  ensures
    bp_sharing m1 (ematrix_subtile eA bm bk mrow bkIdx) nthr **
    bp_sharing m2 (ematrix_subtile eB bk bn bkIdx mcol) nthr

(* Per-thread helpers for even iterations. *)
ghost
fn fold_barrier_p_even
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    (exists* em1. bp_sharing m1 em1 nthr) **
    (exists* em2. bp_sharing m2 em2 nthr)
  ensures
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx) tid

ghost
fn unfold_barrier_q_even
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx) tid
  ensures
    live_strided_chunks m1 nthr tid **
    live_strided_chunks m2 nthr tid
