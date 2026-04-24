module Kuiper.Kernel.GEMM.FlipFlopBarrier2

(* Array2 version of FlipFlopBarrier.
   This module defines a barrier contract used by GEMMs that operate
   on Array2 (Tensor-backed) matrices instead of VArray-backed gpu_matrix. *)

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Tensor.Tiling

open Kuiper.Array2 { array2 }
module M = Kuiper.Array2
module SZ = Kuiper.SizeT
module CV = Kuiper.Kernel.GEMM.Copy.Vec

(* ---- Strided chunk operations for Array2 ---- *)

ghost
fn split_array2_into_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (m : array2 et l)
  (#em : ematrix et rows cols)
  (nthr : pos)
  requires
    m |-> em
  ensures
    pure (SZ.fits (M.layout_size l))
  ensures
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid
{
  M.ilower m;
  forevery_flatten _;
  Classical.forall_intro (CV.in_chunk_covers_all (chunk et #_ #hvc) rows cols nthr);
  forevery_refine_ext #_ #(fun _ -> True)
    (fun (ij : (natlt rows & natlt cols)) ->
      exists tid. CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij)
    _;
  Classical.forall_intro_3 (fun ij tid1 -> Classical.move_requires
                             (CV.in_chunk_no_overlap (chunk et #_ #hvc) rows cols nthr ij tid1));
  forevery_split_or_n _ _;
  ghost
  fn aux (tid : natlt nthr)
    requires
      forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
        M.pts_to_cell m (ij._1, ij._2) (macc em ij._1 ij._2)
    ensures
      own_strided_chunks m em nthr tid
  {
    fold own_strided_chunks m em nthr tid;
  };
  forevery_map _ _ aux;
}

ghost
fn join_array2_from_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (m : array2 et l)
  (#em : ematrix et rows cols)
  (nthr : pos)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid
  ensures
    m |-> em
{
  assert pure (SZ.fits (M.layout_size l));
  forevery_map
    (fun tid -> own_strided_chunks m em nthr tid)
    (fun tid -> forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
        M.pts_to_cell m (ij._1, ij._2) (macc em ij._1 ij._2))
    fn tid { unfold own_strided_chunks m em nthr tid };
  forevery_join_or_n (fun (tid : natlt nthr) ij -> CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij)
    (fun ij -> M.pts_to_cell m (ij._1, ij._2) (macc em ij._1 ij._2));
  Classical.forall_intro (CV.in_chunk_covers_all (chunk et #_ #hvc) rows cols nthr);
  Classical.forall_intro_3 (fun ij tid1 -> Classical.move_requires
                             (CV.in_chunk_no_overlap (chunk et #_ #hvc) rows cols nthr ij tid1));
  forevery_refine_ext #_
    #(fun (ij : (natlt rows & natlt cols)) ->
      exists tid. CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij)
    (fun _ -> True)
    _;
  forevery_unflatten' _;
  M.iraise m;
}

ghost
fn join_array2_from_strided_chunks_underspec
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (m : array2 et l)
  (nthr : pos)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (tid : natlt nthr).
      live_strided_chunks m nthr tid
  ensures
    live m
{
  forevery_map
    (fun (tid : natlt nthr) -> live_strided_chunks m nthr tid)
    (fun (tid : natlt nthr) -> exists* em. own_strided_chunks m em nthr tid)
    fn tid { unfold live_strided_chunks m nthr tid };

  let ff = forevery_exists #(natlt nthr) _;
  let em' : ematrix et rows cols =
    (mkM fun i j ->
       let flat_idx : nat = i * cols + j in
       let chunk_idx = flat_idx / chunk et in
       let tid = chunk_idx % nthr in
       macc (ff tid) i j);

  forevery_map
    (fun (tid : natlt nthr) -> own_strided_chunks m (ff tid) nthr tid)
    (fun (tid : natlt nthr) -> own_strided_chunks m em' nthr tid)
    fn tid {
      unfold own_strided_chunks m (ff tid) nthr tid;
      forevery_map
        #(ij : (natlt rows & natlt cols){CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij})
        (fun ij -> M.pts_to_cell m (ij._1, ij._2) (macc (ff tid) ij._1 ij._2))
        (fun ij -> M.pts_to_cell m (ij._1, ij._2) (macc em' ij._1 ij._2))
        fn ij { () };
      fold own_strided_chunks m em' nthr tid;
    };

  join_array2_from_strided_chunks m nthr;
  assert m |-> em';
}

(* ---- Barrier transform helpers ---- *)

ghost
fn bp_sharing_to_own_strided_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : M.full_layout rows cols)
  (sar : gpu_array et (rows * cols))
  (em : ematrix et rows cols)
  (nthr : pos)
  (#_ : squash (chunk et /?+ cols))
  (#_ : squash (chunk et * nthr /?+ (rows * cols)))
  requires
    forall+ (_tid : natlt nthr).
      bp_sharing (M.from_array l sar) em nthr
  ensures
    forall+ (tid : natlt nthr).
      own_strided_chunks (M.from_array l sar) em nthr tid
{
  M.gather_n (M.from_array l sar) nthr;
  split_array2_into_strided_chunks (M.from_array l sar) nthr;
}

ghost
fn own_strided_chunks_to_bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : M.full_layout rows cols)
  (sar : gpu_array et (rows * cols))
  (em : ematrix et rows cols)
  (nthr : pos)
  (#_ : squash (SZ.fits (M.layout_size l)))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks (M.from_array l sar) em nthr tid
  ensures
    forall+ (_tid : natlt nthr).
      bp_sharing (M.from_array l sar) em nthr
{
  join_array2_from_strided_chunks (M.from_array l sar) nthr;
  M.share_n (M.from_array l sar) nthr;
}

ghost
fn bp_sharing_to_own_strided_chunks_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : M.full_layout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (chunk et /?+ cols))
  (#_ : squash (chunk et * nthr /?+ (rows * cols)))
  requires
    forall+ (_tid : natlt nthr).
      exists* em.
        bp_sharing (M.from_array l sar) em nthr
  ensures
    forall+ (tid : natlt nthr).
      exists* em.
        own_strided_chunks (M.from_array l sar) em nthr tid
{
  M.gather_n_underspec (M.from_array l sar) nthr;
  with em. assert M.from_array l sar |-> em;
  split_array2_into_strided_chunks (M.from_array l sar) nthr;
  forevery_map
    (fun tid -> own_strided_chunks (M.from_array l sar) em nthr tid)
    (fun tid -> exists* em. own_strided_chunks (M.from_array l sar) em nthr tid)
    fn tid { };
}

ghost
fn own_strided_chunks_to_bp_sharing_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : M.full_layout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (SZ.fits (M.layout_size l)))
  requires
    forall+ (tid : natlt nthr).
      exists* em.
        own_strided_chunks (M.from_array l sar) em nthr tid
  ensures
    forall+ (_tid : natlt nthr).
      exists* em.
        bp_sharing (M.from_array l sar) em nthr
{
  join_array2_from_strided_chunks_underspec (M.from_array l sar) nthr;
  with em. assert M.from_array l sar |-> em;
  M.share_n (M.from_array l sar) nthr;
  forevery_map
    (fun (tid : natlt nthr) -> M.from_array l sar |-> Frac (1.0R /. nthr) em)
    (fun (tid : natlt nthr) -> exists* em. bp_sharing (M.from_array l sar) em nthr)
    fn tid { fold bp_sharing (M.from_array l sar) em nthr; };
}

(* ---- Even/odd barrier transforms ---- *)

ghost
fn even_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : M.full_layout bm bk)
  (l2 : M.full_layout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  requires
    forall+ (tid : natlt nthr).
      (exists* em1. bp_sharing (M.from_array l1 sar1) em1 nthr) **
      (exists* em2. bp_sharing (M.from_array l2 sar2) em2 nthr)
  ensures
    forall+ (tid : natlt nthr).
      live_strided_chunks (M.from_array l1 sar1) nthr tid **
      live_strided_chunks (M.from_array l2 sar2) nthr tid
{
  forevery_unzip _ _;
  bp_sharing_to_own_strided_chunks_underspec l1 sar1 nthr;
  bp_sharing_to_own_strided_chunks_underspec l2 sar2 nthr;
  forevery_zip (fun (tid: natlt nthr) ->
      live_strided_chunks (M.from_array l1 sar1) nthr tid) _;
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
  (l1 : M.full_layout bm bk)
  (l2 : M.full_layout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (it : natlt (2 * (shared / bk)))
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (M.layout_size l1)))
  (#_ : squash (SZ.fits (M.layout_size l2)))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
      own_strided_chunks (M.from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid
  ensures
    forall+ (tid : natlt nthr).
      bp_sharing (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
      bp_sharing (M.from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr
{
  forevery_unzip _ _;
  own_strided_chunks_to_bp_sharing l1 sar1 _ nthr;
  own_strided_chunks_to_bp_sharing l2 sar2 _ nthr;
  forevery_zip
    (fun (tid : natlt nthr) ->
      bp_sharing (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr)
      _;
}

(* ---- Main barrier_p_to_q_transform ---- *)

ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : M.full_layout bm bk)
  (l2 : M.full_layout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (M.layout_size l1)))
  (#_ : squash (SZ.fits (M.layout_size l2)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid
  ensures
    forall+ (tid : natlt nthr).
      barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid
{
  if (it >= 2 * (shared / bk)) {
    forevery_map
      (fun (tid : natlt nthr) ->
        barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid)
      (fun (tid : natlt nthr) ->
        barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid)
      fn tid {
        rewrite barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid as emp;
        rewrite emp as barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid;
      };
  } else {
    let ev = even it;
    if ev {
      assert pure (it < 2 * (shared / bk));
      assert pure (even it);
      forevery_map
        (fun (tid : natlt nthr) ->
          barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid)
        (fun (tid : natlt nthr) ->
          (exists* em1. bp_sharing (M.from_array l1 sar1) em1 nthr) **
          (exists* em2. bp_sharing (M.from_array l2 sar2) em2 nthr)
        )
        fn tid {
          rewrite barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid
              as (exists* em1. bp_sharing (M.from_array l1 sar1) em1 nthr) **
                  (exists* em2. bp_sharing (M.from_array l2 sar2) em2 nthr);
        };

      even_barrier_p_to_q eA eB l1 l2 sar1 sar2 nthr;

      forevery_map
        (fun (tid : natlt nthr) ->
          live_strided_chunks (M.from_array l1 sar1) nthr tid **
          live_strided_chunks (M.from_array l2 sar2) nthr tid)
        (fun (tid : natlt nthr) ->
          barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid
        )
        fn tid {
          rewrite
            live_strided_chunks (M.from_array l1 sar1) nthr tid **
            live_strided_chunks (M.from_array l2 sar2) nthr tid
          as
            barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid;
        };
    } else {
      assert pure (it < 2 * (shared / bk));
      assert pure (odd it);
      forevery_map
        (fun (tid : natlt nthr) ->
          barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid)
        (fun (tid : natlt nthr) ->
          own_strided_chunks (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
          own_strided_chunks (M.from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid
        )
        fn tid {
          rewrite
            barrier_p eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid
          as
            own_strided_chunks (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
            own_strided_chunks (M.from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid;
        };

      odd_barrier_p_to_q eA eB l1 l2 sar1 sar2 nthr bid it;

      forevery_map
        (fun (tid : natlt nthr) ->
          bp_sharing (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
          bp_sharing (M.from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr)
        (fun (tid : natlt nthr) ->
          barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid
        )
        fn tid {
          rewrite
            bp_sharing (M.from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
            bp_sharing (M.from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr
          as
            barrier_q eA eB (M.from_array l1 sar1) (M.from_array l2 sar2) nthr bid it tid;
        };
    }
  }
}

(* ---- Per-thread fold/unfold helpers ---- *)

#push-options "--fuel 2"
ghost
fn fold_barrier_p_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : M.layout bm bk)
  (#l2 : M.layout bk bn)
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
  (#l1 : M.layout bm bk)
  (#l2 : M.layout bk bn)
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
  (#l1 : M.layout bm bk)
  (#l2 : M.layout bk bn)
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
  (#l1 : M.layout bm bk)
  (#l2 : M.layout bk bn)
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
{
  rewrite
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx) tid
  as
    live_strided_chunks m1 nthr tid **
    live_strided_chunks m2 nthr tid;
}
#pop-options
