module Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

#lang-pulse

(* This file is really awful in some places, and it shows that
we didn't structure the pre/post conditions as well as we could have.
Or, at least, we didn't take advantage of it. We should do a round later
and try to make these proofs more compositional and uniform.

There is really nothing too fancy going on... but it still takes >1000
lines. *)

#set-options "--z3rlimit 40"

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Bijection
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec
open Kuiper.TensorCore
open Kuiper.VArray { varray, varray_pts_to, varray_pts_to_cell }
open Pulse.Lib.Array
open Pulse.Lib.Trade

module B = Kuiper.Barrier
module R = Kuiper.Matrix.Reprs
module SZ = Kuiper.SizeT

let bid_of_ij
  (rows cols : nat)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (i : natlt rows)
  (j : natlt cols)
  : natlt (rows/bm * (cols/bn))
  = (i / bm) * (cols/bn) + (j / bn)

let wid_of_ij
  (rows cols : nat)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (i : natlt rows)
  (j : natlt cols)
  : natlt (bm/(wm*tm) * (bn/(wn*tn)))
  =
    let i0 : natlt (bm/(wm*tm)) = (i % bm) / (wm*tm) in
    let j0 : natlt (bn/(wn*tn)) = (j % bn) / (wn*tn) in
    i0 * (bn/(wn*tn)) + j0

// probably exists already
let silly_modulo_helper (p : pos)
  (a : nat) (b : natlt p)
  : Lemma ((a * p + b) % p == b)
  = ()
let silly_div_helper (p : pos)
  (a : nat) (b : natlt p)
  : Lemma ((a * p + b) / p == a)
  = ()

let lem_i
  (rows shared cols : nat)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (i : natlt rows)
  (j : natlt cols)
  : Lemma (ensures warp_tile_i #rows #cols bm bn bk tm tn tk wm wn
                       nthr
                       (bid_of_ij rows cols bm bn i j)
                       (wid_of_ij rows cols bm bn tm tn wm wn i j)
                     * (wm*tm) + ((i % bm) % (wm*tm)) == i)
  = let bid = bid_of_ij rows cols bm bn i j in
    let wid = wid_of_ij rows cols bm bn tm tn wm wn i j in
    assert (j/bn < cols/bn);
    silly_modulo_helper (cols/bn) (i/bm) (j/bn);
    assert (bid / (cols/bn) == i / bm);
    silly_div_helper (bn/(wn*tn)) ((i%bm)/(wm*tm)) ((j%bn)/(wn*tn));
    assert (wid / (bn/(wn*tn)) == (i % bm) / (wm*tm));
    calc (==) {
      warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid * (wm*tm)
        + ((i % bm) % (wm*tm));
      == {}
      ((bid / (cols/bn)) * (bm / (wm*tm)) + wid / (bn/(wn*tn))) * (wm*tm)
        + ((i % bm) % (wm*tm));
      == {}
      ((i/bm) * (bm / (wm*tm)) + wid / (bn/(wn*tn))) * (wm*tm)
        + ((i % bm) % (wm*tm));
      == {}
      ((i/bm) * (bm / (wm*tm)) + (i % bm) / (wm*tm)) * (wm*tm)
        + ((i % bm) % (wm*tm));
      == {}
      (i/bm) * (bm / (wm*tm)) * (wm*tm) + ((i % bm) / (wm*tm)) * (wm*tm)
        + ((i % bm) % (wm*tm));
      == { Math.Lemmas.euclidean_division_definition (i % bm) (wm*tm) }
      (i/bm) * (bm / (wm*tm)) * (wm*tm) + (i % bm);
      == {}
      i;
    };
    ()

let lem_j
  (rows shared cols : nat)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (i : natlt rows)
  (j : natlt cols)
  : Lemma (ensures warp_tile_j #rows #cols bm bn bk tm tn tk wm wn
                       nthr
                       (bid_of_ij rows cols bm bn i j)
                       (wid_of_ij rows cols bm bn tm tn wm wn i j)
                     * (wn*tn) + ((j % bn) % (wn*tn)) == j)
  = let bid = bid_of_ij rows cols bm bn i j in
    let wid = wid_of_ij rows cols bm bn tm tn wm wn i j in
    assert (j/bn < cols/bn);
    silly_modulo_helper (cols/bn) (i/bm) (j/bn);
    assert (bid % (cols/bn) == j / bn);
    silly_div_helper (bn/(wn*tn)) ((i%bm)/(wm*tm)) ((j%bn)/(wn*tn));
    assert (wid % (bn/(wn*tn)) == (j % bn) / (wn*tn));
    calc (==) {
      warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid * (wn*tn)
        + ((j % bn) % (wn*tn));
      == {}
      ((bid % (cols/bn)) * (bn/(wn*tn)) + wid % (bn/(wn*tn))) * (wn*tn)
        + ((j % bn) % (wn*tn));
      == {}
      ((j/bn) * (bn/(wn*tn)) + wid % (bn/(wn*tn))) * (wn*tn)
        + ((j % bn) % (wn*tn));
      == {}
      ((j/bn) * (bn/(wn*tn)) + (j % bn) / (wn*tn)) * (wn*tn)
        + ((j % bn) % (wn*tn));
      == {}
      (j/bn) * (bn/(wn*tn)) * (wn*tn) + ((j % bn) / (wn*tn)) * (wn*tn)
        + ((j % bn) % (wn*tn));
      == { Math.Lemmas.euclidean_division_definition (j % bn) (wn*tn) }
      (j/bn) * (bn/(wn*tn)) * (wn*tn) + (j % bn);
      == {}
      j;
    };
    ()

ghost
fn gpu_slice_gather_underspec
  (#a : Type u#0)
  (#sz : nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (m n : nat)
  (k : nat { k > 0 })
  requires
    forall+ (_ : natlt k).
      exists* v. gpu_pts_to_slice arr #(f /. k) m n v
  ensures
    exists* v.
      gpu_pts_to_slice arr #f m n v
{
  forevery_natlt_pop k _;
  with vv. assert gpu_pts_to_slice arr #(f /. k) m n vv;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      gpu_pts_to_slice arr #(f /. k) m n vv ** (exists* v. gpu_pts_to_slice arr #(f /. k) m n v)
    ensures
      gpu_pts_to_slice arr #(f /. k) m n vv ** gpu_pts_to_slice arr #(f /. k) m n vv
  {
    gpu_slice_pts_to_eq arr m n (f /. k) #_ #vv;
  };
  forevery_map_extra #(natlt (k-1)) (gpu_pts_to_slice arr #(f /. k) m n vv)
    (fun (_ : natlt (k-1)) -> exists* v. gpu_pts_to_slice arr #(f /. k) m n v)
    (fun (_ : natlt (k-1)) -> gpu_pts_to_slice arr #(f /. k) m n vv)
    aux;
  forevery_natlt_push k _;
  gpu_slice_gather arr m n k;
}

ghost
fn gpu_matrix_share_threads
  (#et : Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  (nblk nthr : pos)
requires
  gm |-> Frac f em
ensures
  forall+ (bid : natlt nblk) (tid : natlt nthr). gm |-> Frac (f/.(nblk*nthr)) em
{
  gpu_matrix_share_n gm (nblk*nthr);
  forevery_factor (nblk * nthr) nblk nthr _;
}

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /? rows))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /? cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** pure (eA %~ rA) **
    gB |-> Frac fB eB ** pure (eB %~ rB) **
    gC |-> eC ** pure (eC %~ rC)
  ensures
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) **
    pure (SZ.fits (mlayout_size lC)) // frame
{
  gpu_matrix_pts_to_ref gC;
  gpu_matrix_share_threads gA nblk nthr;
  gpu_matrix_share_threads gB nblk nthr;

  gpu_matrix_tile gC bm bn;
  forevery_unfactor' nblk (rows/bm) (cols/bn) _;

  ghost
  fn create_warp_tiles_shared
    (#et : Type0) {| scalar et |}
    (#rows #cols : nat)
    (#l : mlayout rows cols)
    ([@@@mkey] gm : gpu_matrix et l)
    (#f : perm)
    (#em : ematrix et rows cols)
    (trows : nat{trows > 0 /\ trows /? rows})
    (tcols : nat{tcols > 0 /\ tcols /? cols})
    (nthr : nat{nthr == rows/trows * (cols/tcols) * warp_size})
  requires
    gm |-> Frac f em
  ensures
    forall+ (trc : natlt nthr).
      warp_tile gm trows tcols (trc/warp_size)
        |-> Frac (f /. warp_size)
      (ematrix_subtile em trows tcols
        (warp_tile_idx_rows rows cols trows tcols (trc/warp_size))
        (warp_tile_idx_cols rows cols trows tcols (trc/warp_size)))
  {
    gpu_matrix_tile gm trows tcols;
    forevery_unfactor' (rows/trows * (cols/tcols)) (rows/trows) (cols/tcols) _;

    forevery_map
      (fun (trc : natlt (rows/trows * (cols/tcols))) ->
        gpu_matrix_subtile gm trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))
          |-> Frac f (ematrix_subtile em trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))))
      (fun trc ->
        forall+ (_lid: natlt warp_size).
          gpu_matrix_subtile gm trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))
            |-> Frac (f /. warp_size) (ematrix_subtile em trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))))
      fn trc { gpu_matrix_share_n _ warp_size };
    forevery_unfactor' nthr (rows / trows * (cols / tcols)) 32 _;
    ();
  };
  forevery_map
    (fun (trc : natlt nblk) ->
      (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (trc/(cols/bn)) (trc%(cols/bn)))
        // Explicit fraction required, otherwise tactic to resolve it fails?!?!
        |-> Frac 1.0R
      (ematrix_subtile eC bm bn (trc/(cols/bn)) (trc%(cols/bn))))
    _
    (fun trc ->
      create_warp_tiles_shared
        (block_tile gC (SZ.v bm) (SZ.v bn) trc)
        // (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (trc/(cols/bn)) (trc%(cols/bn)))
        (wm*tm)
        (wn*tn)
        nthr);

  forevery_zip_2 #(natlt nblk) #(natlt nthr)
    (fun bid -> fun tid -> gB |-> Frac (fB /. (nblk*nthr)) eB)
    (fun bid -> fun tid ->
      (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) (tid/warp_size))
        |-> Frac (recip warp_size)
      (ematrix_subtile (ematrix_subtile eC bm bn (bid/(cols/bn)) (bid%(cols/bn)))
        (wm*tm) (wn*tn)
        (warp_tile_idx_rows (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
        (warp_tile_idx_cols (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))));

  forevery_zip_2 #(natlt nblk) #(natlt nthr)
    (fun bid -> fun tid -> gA |-> Frac (fA /. (nblk*nthr)) eA)
    _;

  // is this necessary? :/
  ghost
  fn aux (bid : natlt nblk) (tid : natlt nthr)
  requires
    gA |-> Frac (fA /. (nblk*nthr)) eA ** gB |-> Frac (fB /. (nblk*nthr)) eB **
    (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) (tid/warp_size))
      |-> Frac (recip warp_size)
    (ematrix_subtile (ematrix_subtile eC bm bn (bid/(cols/bn)) (bid%(cols/bn)))
      (wm*tm) (wn*tn)
      (warp_tile_idx_rows (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
      (warp_tile_idx_cols (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size)))
  ensures
    kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid
  {
    with tC.
      fold warp_tile_pts_to gC bm bn tm tn wm wn bid (tid/warp_size) tC;
  };
  forevery_map_2 _ _ aux;
  ()
}

ghost
fn bp_sharing_to_bp_exclusive
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
      bp_exclusive (from_array l sar) em nthr tid
{
  gpu_matrix_gather_n (from_array l sar) nthr;
  split_matrix_into_strided_chunks (from_array l sar) nthr;
  forevery_map
    (fun tid -> own_strided_chunks (from_array l sar) em nthr tid)
    (fun tid -> bp_exclusive (from_array l sar) em nthr tid)
    fn tid { fold bp_exclusive (from_array l sar) em nthr tid; };
}

ghost
fn bp_exclusive_to_bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (em : ematrix et rows cols)
  (nthr : pos)
  (#_ : squash (SZ.fits (mlayout_size l)))
  requires
    forall+ (tid : natlt nthr).
      bp_exclusive (from_array l sar) em nthr tid
  ensures
    forall+ (_tid : natlt nthr).
      bp_sharing (from_array l sar) em nthr
{
  join_matrix_from_strided_chunks (from_array l sar) nthr;
  gpu_matrix_share_n (from_array l sar) nthr;
}

ghost
fn bp_sharing_to_bp_exclusive_underspec
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
        bp_exclusive (from_array l sar) em nthr tid
{
  gpu_matrix_gather_n_underspec (from_array l sar) nthr;
  with em. assert from_array l sar |-> em;
  split_matrix_into_strided_chunks (from_array l sar) nthr;
  forevery_map
    (fun tid -> own_strided_chunks (from_array l sar) em nthr tid)
    (fun tid -> exists* em. bp_exclusive (from_array l sar) em nthr tid)
    fn tid { fold bp_exclusive (from_array l sar) em nthr tid; };
}

ghost
fn bp_exclusive_to_bp_sharing_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (SZ.fits (mlayout_size l)))
  requires
    forall+ (tid : natlt nthr).
      exists* em.
        bp_exclusive (from_array l sar) em nthr tid
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
  bp_sharing_to_bp_exclusive_underspec l1 sar1 nthr;
  bp_sharing_to_bp_exclusive_underspec l2 sar2 nthr;
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
      bp_exclusive (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
      bp_exclusive (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid
  ensures
    forall+ (tid : natlt nthr).
      bp_sharing (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr **
      bp_sharing (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr
{
  forevery_unzip _ _;
  bp_exclusive_to_bp_sharing l1 sar1 _ nthr;
  bp_exclusive_to_bp_sharing l2 sar2 _ nthr;
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
          bp_exclusive (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
          bp_exclusive (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid
        )
        fn tid {
          rewrite
            barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
          as
            bp_exclusive (from_array l1 sar1) (ematrix_subtile eA bm bk (bid/(cols/bn)) (it/2)) nthr tid **
            bp_exclusive (from_array l2 sar2) (ematrix_subtile eB bk bn (it/2) (bid%(cols/bn))) nthr tid;
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

ghost
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size /\ nthr <= 1024})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    can_create_barrier nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
  ensures
    consumed_can_create_barrier **
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid) **
    emp (* frame *)
{
  (* permissions for shared memory *)
  // shmem for A tile
  // rewrite each shmems_desc et_ab bm bn bk as [SHArray et_ab (bm *^ bk); SHArray et_ab (bk *^ bn);];
  rewrite live_c_shmems sh as
    live_c_shmem (fst sh) **
    (live_c_shmem (fst (snd sh)) **
     emp);
  rewrite live_c_shmem (fst sh) as (exists* v. gpu_pts_to_array (fst sh) v);
  rewrite live_c_shmem (fst (snd sh)) as (exists* v. gpu_pts_to_array (fst (snd sh)) v);
  gpu_slice_share (fst sh) 0 (bm*bk) nthr;
  gpu_slice_share (fst (snd sh)) 0 (bk*bn) nthr;
  with s1.
    assert (forall+ (x: natlt nthr). gpu_pts_to_slice (fst sh) #(recip nthr) 0 (bm*bk) (reveal s1));
  with s2.
    assert (forall+ (x: natlt nthr). gpu_pts_to_slice (fst (snd sh)) #(recip nthr) 0 (bk*bn) s2);

  ghost fn aux (#n : nat) (arr : gpu_array et_ab n) (s : erased (seq et_ab)) (tid : natlt nthr)
    requires gpu_pts_to_slice arr #(recip nthr) 0 n s
    ensures exists* (x : seq et_ab). gpu_pts_to_array arr #(recip nthr) x
    {};
  forevery_map #(natlt nthr)
    (fun _tid -> gpu_pts_to_slice (fst sh) #(recip nthr) 0 (bm*bk) s1)
    _ (aux (fst sh) s1);
  forevery_map #(natlt nthr)
    (fun _tid -> gpu_pts_to_slice (fst (snd sh)) #(recip nthr) 0 (bk*bn) s2)
    _ (aux (fst (snd sh)) s2);

  (* create barrier token *)
  B.mk_barrier nthr _ _
    (barrier_p_to_q_transform eA eB (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) nthr bid);
  let sar1 = fst sh;
  let sar2 = fst (snd sh);

  (* consolidate permissions under a single forall+ *)
  forevery_zip
    (fun tid -> exists* (x : seq et_ab). gpu_pts_to_array (fst (snd sh)) #(recip nthr) x)
    (fun tid ->
      barrier_tok eA eB (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 nthr bid tid);
  forevery_zip
    (fun tid -> exists* (x : seq et_ab). gpu_pts_to_array (fst sh) #(recip nthr) x)
    (fun tid ->
      (exists* (x : seq et_ab). gpu_pts_to_array (fst (snd sh)) #(recip nthr) x) **
      barrier_tok eA eB (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 nthr bid tid);
  forevery_zip #(natlt nthr)
    (fun tid -> kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) _;
  ()
}

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
{
  forevery_unzip #(natlt nthr)
    (fun tid -> kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
    _;
  forevery_unzip #(natlt nthr)
    (fun _tid -> ((exists* (x: seq et_ab). gpu_pts_to_array (fst sh) #(recip nthr) x)))
    _;
  forevery_unzip #(natlt nthr)
    (fun _tid -> ((exists* (x: seq et_ab). gpu_pts_to_array (fst (snd sh)) #(recip nthr) x)))
    _;

  (* Restore and give back ownership of shared memory arrays. *)
  gpu_slice_gather_underspec (fst sh) 0 (bm*^bk) nthr;
  gpu_slice_gather_underspec (fst (snd sh)) 0 (bk*^bn) nthr;

  assert (exists* v. gpu_pts_to_array (fst sh) v);
  rewrite (exists* v. gpu_pts_to_array (fst sh) v) as live_c_shmem (fst sh);
  rewrite (exists* v. gpu_pts_to_array (fst (snd sh)) v) as live_c_shmem (fst (snd sh));
  rewrite
    live_c_shmem (fst sh) **
    (live_c_shmem (fst (snd sh)) **
     emp)
    as
    live_c_shmems sh;

  (* Drop barrier token. *)
  drop_
    (forall+ (x: natlt nthr).
      barrier_tok eA eB (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) nthr bid x);
  ()
}


ghost
fn warp_tile_pts_to_eq
  (#et : Type0) {| scalar et |}
  (#rows : nat)
  (#cols : nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (em1 em2 : ematrix et (wm * tm) (wn * tn))
  requires
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em1 **
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2
  ensures
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2 **
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2
{
  unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em1;
  unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2;
  gpu_matrix_pts_to_eq
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    (recip warp_size)
    #em1 #em2;
  fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2;
  fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2;
}

ghost
fn warp_tile_pts_to_gatherwarp
  (#et : Type0) {| scalar et |}
  (#rows : nat)
  (#cols : nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : pos{bm /?+ rows})
  (bn : pos{bn /?+ cols})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (em : ematrix et (wm * tm) (wn * tn))
  requires
    forall+ (_ : natlt warp_size).
      warp_tile_pts_to gC bm bn tm tn wm wn bid wid em
  ensures
    warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid em
{
  forevery_map #(natlt warp_size)
    (fun _ -> warp_tile_pts_to gC bm bn tm tn wm wn bid wid em)
    (fun _ ->
      gpu_matrix_pts_to
        (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
        #(recip warp_size)
        em)
    fn _ {
      unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em;
    };
  gpu_matrix_gather_n
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    warp_size;
  fold warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid em;
  ();
}

ghost
fn rhs_is_constant_for_warps_approx
  (#et_ab #et_c : Type0)
  {| scalar et_c |}
  {| real_like et_c |}
  (#rows #shared #cols : pos)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et_c lC)
  (eA : ematrix et_ab rows shared)
  (eB : ematrix et_ab shared cols)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : pos{nblk == rows/bm * (cols/bn)})
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (bid : natlt nblk)
  (emf : natlt nthr -> ematrix et_c (wm*tm) (wn*tn))
  norewrite
  requires
    forall+ (tid : natlt nthr).
      warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) (emf tid) **
      pure (emf tid %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))))
  ensures
    forall+ (wid : natlt (nthr / warp_size)).
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf (wid * warp_size)) **
      pure (emf (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid))))
{
  forevery_factor nthr (nthr / warp_size) warp_size _;
  ghost
  fn aux (wid : natlt (nthr / warp_size))
    requires
      forall+ (i : natlt warp_size).
        warp_tile_pts_to gC bm bn tm tn wm wn bid ((wid * warp_size + i) / warp_size) (emf (wid * warp_size + i)) **
        pure (emf (wid * warp_size + i) %~
          (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)) 0)
                     (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)))))
    ensures
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf (wid * warp_size)) **
      pure (emf (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                  (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid))))
  {
    forevery_natlt_pop_shift _ _;

    rewrite each (wid * warp_size + 0) as (wid * warp_size);
    rewrite each ((wid * warp_size) / warp_size) as wid;
    let em0 = emf (wid * warp_size);
    assert rewrites_to em0 (emf (wid * warp_size));
    assert warp_tile_pts_to gC bm bn tm tn wm wn bid wid em0;
    // ghost
    // fn aux (tid : natlt
    // assert pure (em0 %~
    //   (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)) 0)
    //              (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)))));
    forevery_map_extra
      #(natlt (warp_size - 1))
      (warp_tile_pts_to gC bm bn tm tn wm wn bid wid em0)
      (fun i -> warp_tile_pts_to gC bm bn tm tn wm wn bid ((wid * warp_size + (i+1)) / warp_size) (emf (wid * warp_size + (i+1)))
        ** pure (emf (wid * warp_size + (i+1)) %~
          (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + (i+1)) / warp_size)) 0)
                     (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + (i+1)) / warp_size))))))
      (fun i -> warp_tile_pts_to gC bm bn tm tn wm wn bid wid em0)
      // ^ NB: dropping the pure part above, it doesn't matter as we already have this fact about em0
      fn i {
        rewrite each ((wid * warp_size + (i + 1)) / warp_size) as wid;
        warp_tile_pts_to_eq gC bm bn tm tn wm wn bid wid (emf (wid * warp_size + (i+1))) em0;
        ()
      };

    forevery_natlt_push_shift warp_size
      (fun i -> warp_tile_pts_to gC bm bn tm tn wm wn bid wid em0);
    warp_tile_pts_to_gatherwarp gC bm bn tm tn wm wn bid wid em0;
    ();
  };
  forevery_map _ _ aux;
}

let silly_helper_natlt_prod
  (p q : nat)
  (x : natlt p)
  (y : natlt q)
  : Lemma (ensures x * q + y < p * q)
  = ()

#push-options "--split_queries always" // Would be nice to avoid
let tiles_approx_lemma
  (#et_c : Type0)
  {| scalar et_c |}
  {| real_like et_c |}
  (#rows #shared #cols : pos)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : pos{nblk == rows/bm * (cols/bn)})
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (eC' : ematrix et_c rows cols)
  (emf : natlt nblk -> natlt nthr -> ematrix et_c (wm*tm) (wn*tn))
  : Lemma (requires
            (eC' ==
              ematrix_from_tiles bm bn
                (fun br bc ->
                  ematrix_from_tiles (wm*tm) (wn*tn)
                    (fun tr tc -> emf (br * (cols/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size))))
          /\ (forall (bid : natlt nblk) (wid : natlt (nthr / warp_size)).
                emf bid (wid * warp_size) %~
                  (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                            (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)))))
          (ensures eC' %~ MS.matmul rA rB)
= let aux (i : natlt rows) (j : natlt cols)
    : Lemma (macc eC' i j %~ macc (MS.matmul rA rB) i j)
  =
    let bid : natlt nblk = bid_of_ij rows cols bm bn i j in
    let wid : natlt (nthr / warp_size) = wid_of_ij rows cols bm bn tm tn wm wn i j in
    let ii  : natlt (wm*tm) = (i % bm) % (wm*tm) in
    let jj  : natlt (wn*tn) = (j % bn) % (wn*tn) in
    calc (==) {
      macc eC' i j;
      == {}
      macc (ematrix_from_tiles #_ #rows #cols bm bn
              (fun br bc ->
                ematrix_from_tiles (wm*tm) (wn*tn)
                  (fun tr tc -> emf (br * (cols/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size))))
        i j;
      == {}
      macc (ematrix_from_tiles #_ #bm #bn (wm*tm) (wn*tn)
                  (fun tr tc -> emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size)))
           (i % bm) (j % bn);
      == {}
      macc (emf bid (wid * warp_size)) ii jj;
    };
    assert (macc eC' i j == macc (emf bid (wid * warp_size)) ii jj);
    assert (macc eC' i j %~
      macc
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)))
        ii jj);

    MS.matmul_decompose_lemma rA rB (wm*tm) (wn*tn) (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)
                                                    (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid);
    assert (macc eC' i j %~
      macc
        (ematrix_subtile
          (MS.matmul rA rB)
          (wm*tm) (wn*tn)
          (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)
          (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid))
        ii jj);

    calc (==) {
      macc
        (ematrix_subtile
          (MS.matmul rA rB)
          (wm*tm) (wn*tn)
          (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)
          (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid))
        ii jj;
      == {}
      macc
        (MS.matmul rA rB)
        (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid * (wm*tm) + ii)
        (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid * (wn*tn) + jj);
      == { lem_i rows shared cols bm bn bk tm tn tk wm wn nthr i j;
           lem_j rows shared cols bm bn bk tm tn tk wm wn nthr i j }
      macc (MS.matmul rA rB) i j;
    };

    ()
  in
  Classical.forall_intro_2 aux;
  ()
#pop-options

#push-options "--z3rlimit 60" // the function below is pretty terribly performant
ghost
fn reconstruct_from_warp_approx
  (#et_ab #et_c : Type0)
  {| scalar et_c |}
  {| real_like et_c |}
  (#rows #shared #cols : pos)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et_c lC)
  (eA : ematrix et_ab rows shared)
  (eB : ematrix et_ab shared cols)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : pos{nblk == rows/bm * (cols/bn)})
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  norewrite
  requires
    pure (SZ.fits (mlayout_size lC))
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))
  ensures
    exists* (eC' : ematrix et_c rows cols).
      gC |-> eC' ** pure (eC' %~ MS.matmul rA rB)
{
  (* Expose the existential. *)
  forevery_map_2 #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))))
    (fun bid tid ->
      exists* (em : ematrix et_c (wm*tm) (wn*tn)).
        warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) em **
        pure (em %~
          (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                     (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))))
    fn bid tid {
      unfold warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))));
    };

  (* Choice. *)
  let emf = forevery_exists_2
    (fun (bid : natlt nblk) (tid : natlt nthr) (em : ematrix et_c (wm*tm) (wn*tn)) ->
        warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) em **
        pure (em %~
          (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                     (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))));

  (* Threads within a warp are uniform, factor and gather. *)
  forevery_map #(natlt nblk)
    (fun bid -> forall+ (tid : natlt nthr).
      warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) (emf bid tid) **
      pure (emf bid tid %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))))
    (fun bid -> forall+ (wid : natlt (nthr / warp_size)).
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)) **
      pure (emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)))))
    fn bid {
      rhs_is_constant_for_warps_approx gC eA eB eC bm bn bk tm tn tk wm wn nblk nthr rA rB rC bid (emf bid);
    };

  (* Now reconstruct the full matrix from the big tiles. *)

  (* First, extract all the pure facts at once. *)
  forevery_extract_pure_2
    #(natlt nblk) #(natlt (nthr / warp_size))
    (fun bid wid ->
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)) **
      pure (emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)))))
    (fun bid wid ->
      emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid))))
    fn bid wid {
      ();
    };
  assert pure (forall (bid : natlt nblk) (wid : natlt (nthr / warp_size)).
    emf bid (wid * warp_size) %~
      (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                 (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid))));

  (* Now drop the pures. *)
  forevery_map_2
    #(natlt nblk) #(natlt (nthr / warp_size))
    (fun bid wid ->
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)) **
      pure (emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid wid)))))
    (fun bid wid ->
      // warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)))
      // gpu_matrix_pts_to
      //   (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) wid)
      //   (emf bid (wid * warp_size)))
      gpu_matrix_pts_to
        (gpu_matrix_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
          (wm*tm) (wn*tn)
          (wid / (bn/(wn*tn))) (wid % (bn/(wn*tn))))
        (emf bid (wid * warp_size)))
    fn bid wid {
      unfold warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size));
      rewrite each
        (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) wid)
      as
        gpu_matrix_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
          (wm*tm) (wn*tn)
          (wid / (bn/(wn*tn))) (wid % (bn/(wn*tn)));
      ();
    };

  (* First join every block tile *)
  forevery_map
    #(natlt nblk)
    (fun bid -> forall+ (wid : natlt (nthr / warp_size)).
      gpu_matrix_pts_to
        (gpu_matrix_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
          (wm*tm) (wn*tn)
          (wid / (bn/(wn*tn))) (wid % (bn/(wn*tn))))
        (emf bid (wid * warp_size)))
    (fun bid ->
      gpu_matrix_pts_to (block_tile gC (v bm) (v bn) bid)
        (ematrix_from_tiles (wm*tm)
            (wn*tn)
            (fun tr tc -> emf bid ((tr * (bn/(wn*tn)) + tc) * 32))))
    fn bid {
      forevery_factor (nthr / warp_size) (bm/(wm*tm)) (bn/(wn*tn)) _;
      forevery_ext_2 #(natlt (bm/(wm*tm))) #(natlt (bn/(wn*tn)))
        (fun tr tc ->
          gpu_matrix_pts_to
            (gpu_matrix_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
              (wm*tm) (wn*tn)
              ((tr * (bn/(wn*tn)) + tc) / (bn/(wn*tn))) ((tr * (bn/(wn*tn)) + tc) % (bn/(wn*tn))))
            (emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size)))
        (fun tr tc ->
          gpu_matrix_pts_to
            (gpu_matrix_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
              (wm*tm) (wn*tn)
              tr tc)
            // (emf bid (tr * (bn/(wn*tn)) + tc)))
            (emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size)))
        ;
      gpu_matrix_untile' (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn)
        (fun tr tc ->
          emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size));
    };

  (* Now that we have each block tile, again shuffle them and join. *)
  assert (forall+ (bid : natlt nblk).
      gpu_matrix_pts_to (block_tile gC (v bm) (v bn) bid)
        (ematrix_from_tiles (wm*tm)
            (wn*tn)
            (fun tr tc -> emf bid ((tr * (bn/(wn*tn)) + tc) * 32))));
  forevery_factor nblk (rows/bm) (cols/bn) _;
  forevery_map_2 #(natlt (rows/bm)) #(natlt (cols/bn))
    (fun br bc ->
      gpu_matrix_pts_to (block_tile gC (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc))
        (ematrix_from_tiles (wm*tm) (wn*tn)
            (fun tr tc -> emf (br * (cols/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * 32))))
    (fun br bc ->
      gpu_matrix_pts_to
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc)
        (ematrix_from_tiles (wm*tm) (wn*tn)
            (fun tr tc -> emf (br * (cols/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * 32))))
    fn br bc {
      rewrite each
        block_tile gC (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc)
      as
        gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn)
          (block_tile_idx_rows rows cols (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc))
          (block_tile_idx_cols rows cols (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc));
      assert pure
        (block_tile_idx_rows rows cols (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc)
        ==
        (br * (cols/bn) + bc) / (cols/bn)
      );
      assert pure ((br * (cols/bn) + bc) / (cols/bn) == br);
      rewrite each
        block_tile_idx_rows rows cols (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc)
      as
        br;
      assert pure (
        block_tile_idx_cols rows cols (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc)
        ==
        (br * (cols/bn) + bc) % (cols/bn)
      );
      assert pure ((br * (cols/bn) + bc) % (cols/bn) == bc);
      rewrite each
        block_tile_idx_cols rows cols (SZ.v bm) (SZ.v bn) (br * (cols/bn) + bc)
      as
        bc;

      // These are needed to change the arguments to the implicit layout of gC...
      rewrite each ((br * (cols/bn) + bc) / (cols/bn)) as br;
      rewrite each ((br * (cols/bn) + bc) % (cols/bn)) as bc;

      assert
        gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) br bc |->
          ematrix_from_tiles (wm*tm) (wn*tn)
            (fun tr tc -> emf (br * (cols/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size));

      ();
    };

  gpu_matrix_untile' gC bm bn
    (fun br bc ->
      ematrix_from_tiles (wm*tm) (wn*tn)
        (fun tr tc -> emf (br * (cols/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size)));

  (* We now have full permission of gC. We now need to prove the
     functional post. *)
  with eC'. assert gC |-> eC';

  tiles_approx_lemma bm bn bk tm tn tk wm wn nblk nthr rA rB eC' emf;

  assert pure (eC' %~ MS.matmul rA rB);
  ();
}
#pop-options

ghost
fn teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) **
    pure (SZ.fits (mlayout_size lC)) // frame
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et_c rows cols).
      gC |-> eC' ** pure (eC' %~ MS.matmul rA rB))
{
  forevery_unfactor' (rows/bm * (cols/bn) * nthr) nblk nthr _;
  forevery_unzip _ _;
  forevery_unzip _ _;

  gpu_matrix_gather_n gA (rows/bm * (cols/bn) * nthr);
  gpu_matrix_gather_n gB (rows/bm * (cols/bn) * nthr);

  (* Done with gA and gB. The tricky bit is getting back gC and
  proving it approximates the matmul. *)

  assert forall+ (x : natlt (rows / bm * (cols / bn) * nthr)).
    warp_tile_approximates gC bm bn tm tn wm wn (x / nthr) (x % nthr / warp_size)
      (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i bm bn bk tm tn tk wm wn nthr (x / nthr) (x % nthr / warp_size)) 0)
                 (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j bm bn bk tm tn tk wm wn nthr (x / nthr) (x % nthr / warp_size))));

  forevery_factor'
    (rows/bm * (cols/bn) * nthr)
    (rows/bm * (cols/bn))
    nthr
    (fun (bid : natlt (rows/bm * (cols/bn))) (tid : natlt nthr) ->
      warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                  (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))));

  reconstruct_from_warp_approx
    gC eA eB eC bm bn bk tm tn tk wm wn (rows/bm * (cols/bn)) nthr rA rB rC;

  ();
}

