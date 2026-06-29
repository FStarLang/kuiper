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
  (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
    tensor_pts_to_cell m (idx2 ij._1 ij._2) (macc em ij._1 ij._2)

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
  (em : ematrix et rows cols)
  (nthr : pos)
  : slprop
  = m |-> Frac (1.0R /. nthr) em

let barrier_p
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 et l1)
  (m2 : array2 et l2)
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
      own_strided_chunks m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr tid **
      own_strided_chunks m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr tid

let barrier_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 et l1)
  (m2 : array2 et l2)
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
      bp_sharing m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr **
      bp_sharing m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr

let contract
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray et (bm * bk))
  (sar2 : larray et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.contract nthr =
{
  B.rin  = barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
  B.rout = barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
}

let barrier_tok
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray et (bm * bk))
  (sar2 : larray et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : slprop
  = B.barrier_tok (contract eA eB l1 l2 sar1 sar2 nthr bid)

(* The proof of correctness. *)
ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray et (bm * bk))
  (sar2 : larray et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
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
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 et l1)
  (m2 : array2 et l2)
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
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 et l1)
  (m2 : array2 et l2)
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
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 et l1)
  (m2 : array2 et l2)
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
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 et l1)
  (m2 : array2 et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx) tid
  ensures
    live_strided_chunks m1 nthr tid **
    live_strided_chunks m2 nthr tid
