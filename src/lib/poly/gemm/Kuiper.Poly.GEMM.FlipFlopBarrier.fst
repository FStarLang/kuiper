module Kuiper.Poly.GEMM.FlipFlopBarrier

(* This module defines a barrier contract used by several GEMMs. *)

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix
open Kuiper.Poly.GEMM.Copy.Vec { own_strided_chunks, live_strided_chunks }
open Kuiper.Math { even, odd }
open Kuiper.Matrix.Tiling

open Kuiper.Poly.GEMM.Copy.Vec

ghost
fn bp_sharing_to_own_strided_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (em : ematrix et rows cols)
  (nthr : pos)
  (#_ : squash (chunk et /?+ cols))
  (#_ : squash (chunk et * nthr /?+ (rows * cols)))
  requires
    forall+ (_tid : natlt nthr).
      bp_sharing (from_array l sar) em nthr
  ensures
    forall+ (tid : natlt nthr).
      own_strided_chunks (from_array l sar) em nthr tid
{
  gpu_matrix_gather_n (from_array l sar) nthr;
  split_matrix_into_strided_chunks (from_array l sar) nthr;
}

ghost
fn own_strided_chunks_to_bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (em : ematrix et rows cols)
  (nthr : pos)
  (#_ : squash (SZ.fits (mlayout_size l)))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks (from_array l sar) em nthr tid
  ensures
    forall+ (_tid : natlt nthr).
      bp_sharing (from_array l sar) em nthr
{
  join_matrix_from_strided_chunks (from_array l sar) nthr;
  gpu_matrix_share_n (from_array l sar) nthr;
}

ghost
fn bp_sharing_to_own_strided_chunks_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (chunk et /?+ cols))
  (#_ : squash (chunk et * nthr /?+ (rows * cols)))
  requires
    forall+ (_tid : natlt nthr).
      exists* em.
        bp_sharing (from_array l sar) em nthr
  ensures
    forall+ (tid : natlt nthr).
      exists* em.
        own_strided_chunks (from_array l sar) em nthr tid
{
  gpu_matrix_gather_n_underspec (from_array l sar) nthr;
  with em. assert from_array l sar |-> em;
  split_matrix_into_strided_chunks (from_array l sar) nthr;
  forevery_map
    (fun tid -> own_strided_chunks (from_array l sar) em nthr tid)
    (fun tid -> exists* em. own_strided_chunks (from_array l sar) em nthr tid)
    fn tid { };
}

ghost
fn own_strided_chunks_to_bp_sharing_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (SZ.fits (mlayout_size l)))
  requires
    forall+ (tid : natlt nthr).
      exists* em.
        own_strided_chunks (from_array l sar) em nthr tid
  ensures
    forall+ (_tid : natlt nthr).
      exists* em.
        bp_sharing (from_array l sar) em nthr
{
  join_matrix_from_strided_chunks_underspec (from_array l sar) nthr;
  with em. assert from_array l sar |-> em;
  gpu_matrix_share_n (from_array l sar) nthr;
  forevery_map
    (fun (tid : natlt nthr) -> from_array l sar |-> Frac (1.0R /. nthr) em)
    (fun (tid : natlt nthr) -> exists* em. bp_sharing (from_array l sar) em nthr)
    fn tid { fold bp_sharing (from_array l sar) em nthr; };
}

ghost
fn even_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  requires
    forall+ (tid : natlt nthr).
      (exists* em1. bp_sharing (from_array l1 sar1) em1 nthr) **
      (exists* em2. bp_sharing (from_array l2 sar2) em2 nthr)
  ensures
    forall+ (tid : natlt nthr).
      live_strided_chunks (from_array l1 sar1) nthr tid **
      live_strided_chunks (from_array l2 sar2) nthr tid
{
  forevery_unzip _ _;
  bp_sharing_to_own_strided_chunks_underspec l1 sar1 nthr;
  bp_sharing_to_own_strided_chunks_underspec l2 sar2 nthr;
  forevery_zip (fun (tid: natlt nthr) ->
      live_strided_chunks (from_array l1 sar1) nthr tid) _;
}

ghost
fn odd_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (it : natlt (2 * (shared / bk)))
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (mlayout_size l1)))
  (#_ : squash (SZ.fits (mlayout_size l2)))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
      own_strided_chunks (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid
  ensures
    forall+ (tid : natlt nthr).
      bp_sharing (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
      bp_sharing (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr
{
  forevery_unzip _ _;
  own_strided_chunks_to_bp_sharing l1 sar1 _ nthr;
  own_strided_chunks_to_bp_sharing l2 sar2 _ nthr;
  forevery_zip
    (fun (tid : natlt nthr) ->
      bp_sharing (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr)
      _;
}

ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (mlayout_size l1)))
  (#_ : squash (SZ.fits (mlayout_size l2)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
  ensures
    forall+ (tid : natlt nthr).
      barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
{
  if (it >= 2 * (shared / bk)) {
    forevery_map
      (fun (tid : natlt nthr) ->
        barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid)
      (fun (tid : natlt nthr) ->
        barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid)
      fn tid {
        rewrite barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid as emp;
        rewrite emp as barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid;
      };
  } else {
    // if requires ANF
    let ev = even it;
    if ev {
      assert pure (it < 2 * (shared / bk));
      assert pure (even it);
      forevery_map
        (fun (tid : natlt nthr) ->
          barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid)
        (fun (tid : natlt nthr) ->
          (exists* em1. bp_sharing (from_array l1 sar1) em1 nthr) **
          (exists* em2. bp_sharing (from_array l2 sar2) em2 nthr)
        )
        fn tid {
          rewrite barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
              as (exists* em1. bp_sharing (from_array l1 sar1) em1 nthr) **
                  (exists* em2. bp_sharing (from_array l2 sar2) em2 nthr);
        };

      even_barrier_p_to_q eA eB l1 l2 sar1 sar2 nthr;

      forevery_map
        (fun (tid : natlt nthr) ->
          live_strided_chunks (from_array l1 sar1) nthr tid **
          live_strided_chunks (from_array l2 sar2) nthr tid)
        (fun (tid : natlt nthr) ->
          barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
        )
        fn tid {
          rewrite
            live_strided_chunks (from_array l1 sar1) nthr tid **
            live_strided_chunks (from_array l2 sar2) nthr tid
          as
            barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid;
        };
    } else {
      assert pure (it < 2 * (shared / bk));
      assert pure (odd it);
      forevery_map
        (fun (tid : natlt nthr) ->
          barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid)
        (fun (tid : natlt nthr) ->
          own_strided_chunks (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
          own_strided_chunks (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid
        )
        fn tid {
          rewrite
            barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
          as
            own_strided_chunks (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
            own_strided_chunks (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid;
        };

      odd_barrier_p_to_q eA eB l1 l2 sar1 sar2 nthr bid it;

      forevery_map
        (fun (tid : natlt nthr) ->
          bp_sharing (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
          bp_sharing (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr)
        (fun (tid : natlt nthr) ->
          barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
        )
        fn tid {
          rewrite
            bp_sharing (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
            bp_sharing (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr
          as
            barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid;
        };
    }
  }
}

#push-options "--fuel 2 --z3rlimit 20"
ghost
fn fold_barrier_p_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
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
{
  rewrite
    own_strided_chunks m1 (ematrix_subtile eA bm bk mrow bkIdx) nthr tid **
    own_strided_chunks m2 (ematrix_subtile eB bk bn bkIdx mcol) nthr tid
  as
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx + 1) tid;
}
#pop-options

#push-options "--fuel 2"
ghost
fn unfold_barrier_q_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
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
{
  rewrite
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx + 1) tid
  as
    bp_sharing m1 (ematrix_subtile eA bm bk mrow bkIdx) nthr **
    bp_sharing m2 (ematrix_subtile eB bk bn bkIdx mcol) nthr;
}
#pop-options

#push-options "--fuel 2"
ghost
fn fold_barrier_p_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    (exists* em1. bp_sharing m1 em1 nthr) **
    (exists* em2. bp_sharing m2 em2 nthr)
  ensures
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx) tid
{
  rewrite
    (exists* em1. bp_sharing m1 em1 nthr) **
    (exists* em2. bp_sharing m2 em2 nthr)
  as
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx) tid;
}
#pop-options

#push-options "--fuel 2"
ghost
fn unfold_barrier_q_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx) tid
  ensures
    live_strided_chunks m1 nthr tid **
    live_strided_chunks m2 nthr tid
{
  rewrite
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx) tid
  as
    live_strided_chunks m1 nthr tid **
    live_strided_chunks m2 nthr tid;
}
#pop-options
