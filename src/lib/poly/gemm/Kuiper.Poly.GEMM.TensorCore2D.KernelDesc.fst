module Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

#lang-pulse

#set-options "--z3rlimit 40"

open Kuiper

open Kuiper.Matrix.Reprs
module R = Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Matrix

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

open Kuiper.Matrix.Reprs
module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix
open Kuiper.VArray {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
open Kuiper.TensorCore
open Kuiper.Float16
open Kuiper.Matrix.Tiling

open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec

open Pulse.Lib.Array
open Pulse.Lib.Trade

open Kuiper.Bijection

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
    emp (* frame *)
{
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
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost (* comb *) gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
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

  drop_
    (forall+ (x: natlt nthr).
      barrier_tok eA eB (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) nthr bid x);
  ()
}

ghost
fn forevery_factor'
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt d1 -> natlt d2 -> slprop)
  requires
    forall+ (i:natlt n). p (i/d2) (i%d2)
  ensures
    forall+ (i1:natlt d1) (i2:natlt d2). p i1 i2
{
  forevery_factor n d1 d2 (fun i -> p (i / d2) (i % d2));
  forevery_ext_2
    (fun (i1 : natlt d1) (i2 : natlt d2) -> p ((i1 * d2 + i2) / d2) ((i1 * d2 + i2) % d2))
    (fun (i1 : natlt d1) (i2 : natlt d2) -> p i1 i2);
  ();
}

ghost
fn untile_warp_tiles_shared
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#f : perm)
  (trows : nat{trows > 0 /\ trows /? rows})
  (tcols : nat{tcols > 0 /\ tcols /? cols})
  (nthr : nat{nthr == rows/trows * (cols/tcols) * warp_size})
requires
  forall+ (tid : natlt nthr).
    (exists* (em : ematrix _ _ _).
      gpu_matrix_subtile gm trows tcols
        (warp_tile_idx_rows rows cols trows tcols (tid/warp_size))
        (warp_tile_idx_cols rows cols trows tcols (tid/warp_size))
      |-> Frac (f /. warp_size) em)
ensures
  (exists* (em : ematrix _ _ _). gm |-> Frac f em)
{
  forevery_factor nthr (rows/trows * (cols/tcols)) warp_size _;
  ghost
  fn unshare_within_warp (wid : natlt (rows/trows * (cols/tcols)))
  requires
    forall+ (lid: natlt warp_size). (exists* (em : ematrix _ _ _).
        gpu_matrix_subtile gm
              trows
              tcols
              ((wid * 32 + lid) / 32 / (cols / tcols))
              ((wid * 32 + lid) / 32 % (cols / tcols)) |-> Frac (f /. warp_size) em)
  ensures
    (exists* (em : ematrix _ _ _).
      gpu_matrix_subtile gm trows tcols (wid/(cols/tcols)) (wid%(cols/tcols))
        |-> Frac f em)
  {
    forevery_map
      (fun (lid : natlt warp_size) -> exists* (em: ematrix et trows tcols).
        (gpu_matrix_subtile gm trows tcols
              ((wid * 32 + lid) / 32 / (cols / tcols))
              ((wid * 32 + lid) / 32 % (cols / tcols))) |-> Frac (f /. warp_size) em)
      (fun (_lid : natlt warp_size) -> exists* (em: ematrix et trows tcols).
        (gpu_matrix_subtile gm trows tcols
              (wid / (cols / tcols))
              (wid % (cols / tcols)) |-> Frac (f /. warp_size) em))
      fn _lid { rewrite each ((wid * 32 + _lid) / 32) as wid };
    gpu_matrix_gather_n_underspec
      (gpu_matrix_subtile gm trows tcols (wid/(cols/tcols)) (wid%(cols/tcols)))
      warp_size;
  };
  forevery_map _ _ unshare_within_warp;

  forevery_factor' (rows/trows * (cols/tcols)) (rows/trows) (cols/tcols)
    (fun (tr : natlt (rows/trows)) (tc : natlt (cols/tcols)) ->
      exists* (em: ematrix et trows tcols).
          (gpu_matrix_subtile gm trows tcols tr tc) |-> Frac f em);
  gpu_matrix_untile_underspec gm trows tcols;
  ()
}

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
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) **
    emp
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

  forevery_factor'
    (rows/bm * (cols/bn) * nthr)
    (rows/bm * (cols/bn))
    nthr
    (fun (bid : natlt (rows/bm * (cols/bn))) (tid : natlt nthr) ->
      exists* tC.
        warp_tile_pts_to #et_c #_ #(hide #nat (v rows)) #(v cols) #lC gC (v bm)
              (v bn) (v tm) (v tn) (v wm) (v wn)
              bid (tid / 32) tC);

  ghost
  fn aux (bid : natlt (rows/bm * (cols/bn))) (tid : natlt nthr)
    norewrite
    requires
      exists* tC. warp_tile_pts_to gC bm bn tm tn wm wn (bid) (tid/warp_size) tC
    ensures
      exists* tC.
        gpu_matrix_pts_to
          (gpu_matrix_subtile
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn)
              (block_tile_idx_rows (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid)
              (block_tile_idx_cols (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid))
            (wm*tm) (wn*tn)
            (warp_tile_idx_rows (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
            (warp_tile_idx_cols (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
          )
          #(1.0R /. warp_size)
          tC
  {
    with tC. assert warp_tile_pts_to gC bm bn tm tn wm wn bid (tid/warp_size) tC;
    unfold warp_tile_pts_to gC bm bn tm tn wm wn bid (tid/warp_size) tC;
  };
  forevery_map_2 _ _ aux;

  forevery_map
    (fun (bid : natlt ((rows/bm * (cols/bn)))) -> forall+ (tid: natlt nthr).
      exists* (tC: ematrix et_c (wm * tm) (wn * tn)).
        gpu_matrix_subtile
          (gpu_matrix_subtile gC (v bm) (v bn) (bid / (cols/bn)) (bid % (cols/bn)))
              (wm*tm)
              (wn*tn)
              (tid / 32 / (bn / (wn*tn)))
              (tid / 32 % (bn / (wn*tn))) |-> Frac (1.0R /. warp_size) tC)
    _
    (fun (bid : natlt nblk) -> untile_warp_tiles_shared
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn)
          (block_tile_idx_rows (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid)
          (block_tile_idx_cols (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid))
        (wm*tm) (wn*tn) nthr);

  forevery_factor' ((rows/bm) * (cols/bn)) (rows/bm) (cols/bn)
    (fun (trow : natlt (rows/bm)) -> fun (tcol : natlt (cols/bn)) ->
      (exists* (em : ematrix et_c bm bn).
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) trow tcol) |-> em));
  gpu_matrix_untile_underspec gC bm bn;

  with eC'. assert gC |-> eC';
  assume pure (eC' %~ MS.matmul rA rB); // functional correctness admitted

  ()
}
