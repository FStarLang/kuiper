module Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

#lang-pulse

// #set-options "--split_queries always"
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
fn linearize_tile_indices
  (#et : Type0)
  (#rows #cols : nat)
  (trows : nat{trows > 0 /\ trows /? rows})
  (tcols : nat{tcols > 0 /\ tcols /? cols})
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  (n : nat {n == rows/trows * (cols/tcols)})
requires
  forall+ (tr: natlt (rows/trows)) (tc: natlt (cols/tcols)).
    gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
ensures
  forall+ (trc : natlt n).
    gpu_matrix_subtile gm trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))
      |-> Frac f (ematrix_subtile em trows tcols (trc/(cols/tcols)) (trc%(cols/tcols)))
{
  forevery_unfactor' (n) (rows/trows) (cols/tcols) _;
}

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
  linearize_tile_indices trows tcols gm (rows/trows * (cols/tcols));

  ghost
  fn share_within_warp (trc : natlt (rows/trows * (cols/tcols)))
  requires
    gpu_matrix_subtile gm trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))
      |-> Frac f (ematrix_subtile em trows tcols (trc/(cols/tcols)) (trc%(cols/tcols)))
  ensures
    forall+ (_lid: natlt warp_size).
      gpu_matrix_subtile gm trows tcols (trc/(cols/tcols)) (trc%(cols/tcols))
        |-> Frac (f /. warp_size) (ematrix_subtile em trows tcols (trc/(cols/tcols)) (trc%(cols/tcols)))
  {
    gpu_matrix_share_n _ warp_size;
  };
  forevery_map _ _ share_within_warp;
  forevery_unfactor' nthr (rows / trows * (cols / tcols)) 32 _;
  ();
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
  (#_ : squash (bn /? cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid) **
    emp (* frame *)
{
  gpu_matrix_share_threads gA nblk nthr;
  gpu_matrix_share_threads gB nblk nthr;

  gpu_matrix_tile gC bm bn;
  linearize_tile_indices bm bn gC nblk;

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

  forevery_zip_2 #(natlt nblk) #_ #(natlt nthr)
    (fun bid -> fun tid -> gB |-> Frac (fB /. (nblk*nthr)) eB)
    (fun bid -> fun tid ->
      (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) (tid/warp_size))
        |-> Frac (recip warp_size)
      (ematrix_subtile (ematrix_subtile eC bm bn (bid/(cols/bn)) (bid%(cols/bn)))
        (wm*tm) (wn*tn)
        (warp_tile_idx_rows (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
        (warp_tile_idx_cols (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))));

  forevery_zip_2 #(natlt nblk) #_ #(natlt nthr)
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
    kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid
  {
    fold live_warp_tile gC bm bn tm tn wm wn bid (tid/warp_size);
  };
  forevery_map_2 _ _ aux;
  ()
}

#push-options "--debug SMTFail --split_queries always"
let bij_tile_stride_cells (et : Type0) {| sized et, hvc : has_vec_cpy et|} (#nthr : nat) (tid : natlt nthr)
  (rows cols : nat) (#_: squash (chunk et  * nthr /?+ (rows * cols))) (#_: squash (chunk et /?+ cols))
: (natlt (rows*cols/(chunk et * nthr)) =~ idx:natlt (rows*cols) {tid = idx/(chunk et) % nthr /\ chunk et /?+ idx /\ idx%cols < cols - chunk et + 1})
=
{
  ff = admit(); //(fun (i : natlt (rows*cols/(chunk et * nthr))) ->
    //assume (tid = (i * (chunk et * nthr) + tid * chunk et)/(chunk et) % nthr);
    //assume (chunk et /?+ (i * (chunk et * nthr) + tid * chunk et));
    //assume ((i * (chunk et * nthr) + tid * chunk et)%cols < cols - chunk et + 1);
    //i * (chunk et * nthr) + tid * chunk et
    // <: (idx:natlt (rows*cols) {tid = idx/(chunk et) % nthr /\ chunk et /?+ idx /\ idx%cols < cols - chunk et + 1}));
  gg = (fun (i : natlt (rows*cols) {tid = i/(chunk et #_ #hvc) % nthr /\ (chunk et #_ #hvc) /?+ i /\ i%cols < cols - (chunk et #_ #hvc) + 1}) ->
    i/(chunk et * nthr) <: natlt (rows*cols/(chunk et * nthr)));
  ff_gg = ez;
  gg_ff = ez;
}
 
ghost
fn single_even_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (chunk et /?+ cols))
  (#_ : squash (chunk et * nthr /?+ (rows * cols)))
requires
  forall+ (tid : natlt nthr).
    (exists* (x : ematrix _ _ _). (from_array l sar) |-> Frac (1.0R /. nthr) x)
ensures
  forall+ (tid : natlt nthr).
    live_tile_stride_cells (from_array l sar) nthr tid
{
  gpu_matrix_gather_n_underspec (from_array l sar) nthr;
  with em1. assert (from_array l sar) |-> em1;
  gpu_matrix_explode _;
  forevery_unfactor' (rows*cols) rows cols _;

  Kuiper.Divides.lemma_divides_product_r (chunk et) cols rows;
  assert pure (rows * cols / (chunk et) * (chunk et) == rows * cols);
  forevery_factor (rows*cols) (rows*cols/chunk et) (chunk et) _;
  with em'. assert
    forall+ (i : natlt (rows*cols/chunk et)).
      forall+ (k : natlt (chunk et)).
        gpu_matrix_pts_to_cell (from_array l sar)
          ((i * chunk et + k) / cols)
          ((i * chunk et + k) % cols)
          (macc em'
            ((i * chunk et + k) / cols)
            ((i * chunk et + k) % cols));

  (*
  open Pulse.Lib.WithPure;
  ghost
  fn aux_cell (i : natlt (rows*cols/chunk et)) (k : natlt (chunk et))
  requires
    gpu_matrix_pts_to_cell (from_array l sar)
      ((i * chunk et + k) / cols) ((i * chunk et + k) % cols)
      (macc em' ((i * chunk et + k) / cols) ((i * chunk et + k) % cols))
  ensures
      with_pure (i * chunk et % cols + k < cols)
        (fun _ -> live_cell (from_array l sar) (i * chunk et / cols) (i * chunk et % cols + k))
  {
    assert pure (chunk et /?+ cols /\ k < chunk et); 
    assert pure (i * chunk et < rows*cols-chunk et + 1);
    assert pure (i * chunk et + k < (i+1) * chunk et);
    // assert pure ((i * chunk et) % cols < cols - chunk et + 1);
    // assert pure ((i * chunk et + k) % cols = i * chunk et % cols + k);
    // assert pure (exists j. i * chunk et % cols = i * chunk et / cols + j * chunk et);
    assume pure (((i * chunk et + k) / cols) = i * chunk et / cols);
    assume pure (((i * chunk et + k) % cols) = i * chunk et % cols + k);
    rewrite each ((i * chunk et + k) / cols) as (i * chunk et / cols);
    rewrite each ((i * chunk et + k) % cols) as (i * chunk et % cols + k);
    fold live_cell (from_array l sar) (i * chunk et / cols) (i * chunk et % cols +k);
  };
  forevery_map_2 _ _ aux_cell;
  admit();

  ghost
  fn aux_chunk (i : natlt (rows*cols/chunk et))
  requires
    forall+ (k : natlt (chunk et)).
      with_pure (i * chunk et % cols + k < cols)
        (fun _ -> live_cell (from_array l sar) (i * chunk et / cols) (i * chunk et % cols + k))
  ensures
    with_pure (i * chunk et % cols < cols - chunk et + 1)
      (fun _ -> live_chunk (from_array l sar) (i * chunk et / cols) (i * chunk et % cols))
  {
    admit(); 
    fold live_chunk (from_array l sar) (i * chunk et / cols) (i * chunk et % cols);
  };
  admit();
  *)

  forevery_factor (rows*cols/chunk et) nthr (rows*cols/(chunk et * nthr)) _;
  
  with em. assert
    forall+ (tid : natlt nthr).
      forall+ (i : natlt (rows*cols/(chunk et * nthr))).
        forall+ (k : natlt (chunk et)).
          gpu_matrix_pts_to_cell (from_array l sar)
                  (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) / cols)
                  (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) % cols)
                  (macc em
                      (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) / cols)
                      (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) % cols));
  ghost
  fn aux (tid : natlt nthr)
  requires
    forall+ (i : natlt (rows*cols/(chunk et * nthr))).
      forall+ (k : natlt (chunk et)).
        gpu_matrix_pts_to_cell (from_array l sar)
          (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) / cols)
          (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) % cols)
          (macc em
              (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) / cols)
              (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) % cols))
  ensures
    live_tile_stride_cells (from_array l sar) nthr tid
  {
    forevery_iso #(natlt (rows*cols/(chunk et * nthr))) (bij_tile_stride_cells et tid rows cols) _;
    admit();
    ghost
    fn aux2 (idx : natlt (rows*cols) {tid = idx/(chunk et) % nthr /\ chunk et /?+ idx /\ idx%cols < cols - chunk et + 1})
    requires
      forall+ (k : natlt (chunk et)).
        gpu_matrix_pts_to_cell (from_array l sar)
          (((tid * (rows * cols / (chunk et * nthr)) + idx) * chunk et + k) / cols)
          (((tid * (rows * cols / (chunk et * nthr)) + idx) * chunk et + k) % cols)
          (macc em
              (((tid * (rows * cols / (chunk et * nthr)) + idx) * chunk et + k) / cols)
              (((tid * (rows * cols / (chunk et * nthr)) + idx) * chunk et + k) % cols))
    ensures
      live_chunk (from_array l sar) (idx/cols) (idx%cols)
    {
      //ghost
      //fn aux3 (k : natlt (chunk et))
      //requires
      //  gpu_matrix_pts_to_cell (from_array l sar)
      //    (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) / cols)
      //    (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) % cols)
      //    (macc em
      //        (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) / cols)
      //        (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et + k) % cols))
      //ensures live_cell (from_array l sar) (((tid * (rows * cols / (chunk et * nthr)) + i) * chunk et) / cols) ()
      admit();
      fold live_chunk (from_array l sar) (idx/cols) (idx%cols);
    };
    forevery_map _ _ aux2;
    admit();
    ()
  };

  forevery_map _ _ aux;
  ()
}

ghost
fn even_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#bm #bn #bk: szp)
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
    (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (recip nthr) x) **
    (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (recip nthr) x)
ensures
  forall+ (tid : natlt nthr).
    live_tile_stride_cells (from_array l1 sar1) nthr tid **
    live_tile_stride_cells (from_array l2 sar2) nthr tid
{
  forevery_unzip _ _;
  single_even_barrier_p_to_q l1 sar1 nthr;
  single_even_barrier_p_to_q l2 sar2 nthr;
  forevery_zip (fun (tid: natlt nthr) ->
      live_tile_stride_cells (from_array l1 sar1) nthr tid) _;
}

let lemma_divides_prod_divisor (d1 d2 : pos) (a : nat)
  : Lemma (requires (d1 * d2) /? a)
          (ensures d1 /? a /\ d2 /? a)
= admit()

ghost
fn gpu_matrix_implode_underspec
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+ r c.
      exists* v. gpu_matrix_pts_to_cell gm #f r c v
  ensures
    exists* em. gpu_matrix_pts_to gm #f em
{
  admit();
}

ghost
fn single_odd_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp)
  (l : full_mlayout rows cols)
  (sar : gpu_array et (rows * cols))
  (nthr : pos)
  (#_ : squash (chunk et /?+ cols))
  (#_ : squash (chunk et * nthr /?+ (rows * cols)))
requires
  forall+ (tid : natlt nthr). live_tile_stride_cells (from_array l sar) nthr tid
ensures
  forall+ (tid : natlt nthr).
    (exists* (em : ematrix _ _ _). (from_array l sar) |-> Frac (recip nthr) em)
{
  ghost
  fn aux (tid : natlt nthr)
  requires
    live_tile_stride_cells (from_array l sar) nthr tid
  ensures
    forall+ (idx : natlt (rows*cols) {tid = idx/(chunk et) % nthr /\ chunk et /?+ idx /\ idx%cols < cols - chunk et + 1}).
      forall+ (k : natlt (chunk et)).
        live_cell (from_array l sar) (idx/cols) (idx%cols + k)
  {
    unfold live_tile_stride_cells (from_array l sar);

    forevery_iso_back (bij_tile_stride_cells et tid rows cols)
      (fun i -> live_chunk (from_array l sar) ((i * (chunk et * nthr)) / cols) ((i * (chunk et * nthr)) % cols));
    admit();
  };
  forevery_map _ _ aux;
  admit();

  forevery_unfactor' (rows*cols/chunk et) nthr (rows*cols/(chunk et * nthr)) _;
  Kuiper.Divides.lemma_divides_exact (chunk et) (rows*cols);
  lemma_divides_prod_divisor (chunk et) nthr (rows*cols);
  forevery_unfactor' (rows*cols) (rows*cols/chunk et) (chunk et) _;

  forevery_factor (rows*cols) rows cols _;
  admit();
  gpu_matrix_implode_underspec _;
  with em1. assert (from_array l sar) |-> em1;
  gpu_matrix_gather_n_underspec (from_array l sar) nthr;

//  ((((i1 * cols + i2) / chunk /
//                (rows * cols / (chunk * nthr)) *
//                (rows * cols / (chunk * nthr)) +
//                (i1 * cols + i2) /
//                chunk %
//                (rows * cols / (chunk * nthr))) *
//
//              chunk +
//              (i1 * cols + i2) % chunk) /
//            cols)
}

ghost
fn odd_barrier_p_to_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#bm #bn #bk : szp)
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
    live_tile_stride_cells (from_array l1 sar1) nthr tid **
    live_tile_stride_cells (from_array l2 sar2) nthr tid
ensures
  forall+ (tid : natlt nthr).
    (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (recip nthr) x) **
    (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (recip nthr) x)
{
  forevery_unzip _ _;
  single_odd_barrier_p_to_q l1 sar1 nthr;
  single_odd_barrier_p_to_q l2 sar2 nthr;
  forevery_zip
    (fun (tid : natlt nthr) ->
      exists* (em: ematrix et (v bm) (v bk)).
        (from_array l1 sar1) |-> Frac (recip nthr) em) _;
}

ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#bm #bn #bk : szp)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (it : nat)
requires
  forall+ (tid : natlt nthr). barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr it tid
ensures
  forall+ (tid : natlt nthr). barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr it tid
{
  // if requires ANF
  let ev = even it;
  if ev {
    ghost
    fn evaux (tid : natlt nthr)
      norewrite
      requires
        (barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr) it tid
      ensures
        (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (1.0R /. nthr) x) **
        (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (1.0R /. nthr) x)
    {
      rewrite barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr it tid
           as (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (1.0R /. nthr) x) **
              (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (1.0R /. nthr) x);
    };
    forevery_map _ _ evaux;

    even_barrier_p_to_q l1 l2 sar1 sar2 nthr;

    ghost
    fn evaux2 (tid : natlt nthr)
    requires
      live_tile_stride_cells (from_array l1 sar1) nthr tid **
      live_tile_stride_cells (from_array l2 sar2) nthr tid
    ensures
      barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr it tid
    {
      rewrite live_tile_stride_cells (from_array l1 sar1) nthr tid **
              live_tile_stride_cells (from_array l2 sar2) nthr tid
           as barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr it tid;
    };
    forevery_map _ _ evaux2;
  } else {
    ghost
    fn oddaux (tid : natlt nthr)
      norewrite
      requires
        barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr it tid
      ensures
        live_tile_stride_cells (from_array l1 sar1) nthr tid **
        live_tile_stride_cells (from_array l2 sar2) nthr tid
    {
      rewrite barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr it tid
           as live_tile_stride_cells (from_array l1 sar1) nthr tid **
              live_tile_stride_cells (from_array l2 sar2) nthr tid;
    };

    forevery_map _ _ oddaux;
    odd_barrier_p_to_q l1 l2 sar1 sar2 nthr;

    ghost
    fn oddaux2 (tid : natlt nthr)
      norewrite
      requires
        // explicit sizes are not helping
        (exists* (x : ematrix et bm bk). (from_array l1 sar1) |-> Frac (recip nthr) x) **
        (exists* (x : ematrix et bk bn). (from_array l2 sar2) |-> Frac (recip nthr) x)
      ensures
        barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr it tid
    {
      // extra rewrite step reduces time for type checking by ~50%
      rewrite
          (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (recip nthr) x) **
          (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (recip nthr) x)
      as
        (if even (it+1) then
          (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (recip nthr) x) **
          (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (recip nthr) x)
        else
          live_tile_stride_cells (from_array l1 sar1) nthr tid **
          live_tile_stride_cells (from_array l2 sar2) nthr tid);
      rewrite
        (if even (it+1) then
          (exists* (x : ematrix _ _ _). (from_array l1 sar1) |-> Frac (recip nthr) x) **
          (exists* (x : ematrix _ _ _). (from_array l2 sar2) |-> Frac (recip nthr) x)
        else
          live_tile_stride_cells (from_array l1 sar1) nthr tid **
          live_tile_stride_cells (from_array l2 sar2) nthr tid)
      as
        barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr it tid;
    };
    forevery_map _ _ oddaux2;
  }
}

ghost
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
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
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size /\ nthr <= 1024})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    can_create_barrier nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid)
  ensures
    consumed_can_create_barrier **
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr sh bid tid) **
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

  // introduce exists under forall+
  // :( I do not see another way for introducing the exists
  // gpu_pts_to_array (although unfold) does not work because the array sizes are not matched:
  //  there are bm*^bk and bm*bk in the context. while gpu_pts_to_array uses either bm*^bk or bm*bk
  // with s. assert (forall+ (x: natlt nthr). gpu_pts_to_array #_ #(bm*^bk) (fst sh) #(recip nthr) s);
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
  forevery_map
    (fun _tid -> gpu_pts_to_slice (fst sh) #(1.0R /. nthr) 0 (bm*bk) s1) _
    (aux (fst sh) s1);
  forevery_map
    (fun _tid -> gpu_pts_to_slice (fst (snd sh)) #(1.0R /. nthr) 0 (bk*bn) s2) _
    (aux (fst (snd sh)) s2);

  (* create barrier token *)
  B.mk_barrier nthr _ _
    (barrier_p_to_q_transform (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) nthr);
  let sar1 = fst sh;
  let sar2 = fst (snd sh);

  (* consolidate permissions under a single forall+ *)
  forevery_zip
    (fun tid -> exists* (x : seq et_ab). gpu_pts_to_array (fst (snd sh)) #(recip nthr) x)
    (fun tid ->
      barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 nthr tid);
  forevery_zip
    (fun tid -> exists* (x : seq et_ab). gpu_pts_to_array (fst sh) #(recip nthr) x)
    (fun tid ->
      (exists* (x : seq et_ab). gpu_pts_to_array (fst (snd sh)) #(recip nthr) x) **
      barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 nthr tid);
  forevery_zip #(natlt nthr)
    (fun tid -> kpre1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid) _;
  ()
}

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
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
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost (* comb *) gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid)
{
  forevery_unzip #(natlt nthr)
    (fun tid -> kpost1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid)
    _;
  forevery_unzip #(natlt nthr)
    (fun _tid -> ((exists* (x: seq et_ab). gpu_pts_to_array (fst sh) #(recip nthr) x)))
    _;
  forevery_unzip #(natlt nthr)
    (fun _tid -> ((exists* (x: seq et_ab). gpu_pts_to_array (fst (snd sh)) #(recip nthr) x)))
    _;

  // rewrite each Kuiper.Enumerable.cardinal (natlt nthr) #_ as SZ.v nthr;
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
      barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) nthr x);
  ()
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
  admit();
  // forevery_iso #(natlt nthr) (bij_self (natlt (rows/trows * (cols/tcols) * warp_size))) _;
  // forevery_unflatten_natlt
  //   (fun (trc : natlt (rows/trows * (cols/tcols))) -> fun (_lid : natlt warp_size) ->
  //     (exists* (em : ematrix _ _ _).
  //       gpu_matrix_subtile gm trows tcols
  //         (warp_tile_idx_rows rows cols trows tcols (trc))
  //         (warp_tile_idx_cols rows cols trows tcols (trc))
  //     |-> Frac (f /. warp_size) em));

  // ghost
  // fn unshare_within_warp (wid : natlt (rows/trows * (cols/tcols)))
  // requires
  //   forall+ (_lid: natlt warp_size). (exists* (em : ematrix _ _ _).
  //     gpu_matrix_subtile gm trows tcols (wid/(cols/tcols)) (wid%(cols/tcols))
  //       |-> Frac (f /. warp_size) em)
  // ensures
  //   (exists* (em : ematrix _ _ _).
  //     gpu_matrix_subtile gm trows tcols (wid/(cols/tcols)) (wid%(cols/tcols))
  //       |-> Frac f em)
  // {
  //   forevery_tostar (fun (lid : natlt warp_size) ->
  //    (exists* (em : ematrix _ _ _).
  //       gpu_matrix_subtile gm trows tcols (wid/(cols/tcols)) (wid%(cols/tcols))
  //         |-> Frac (f /. warp_size) em));
  //   rewrite each Kuiper.Enumerable.cardinal (natlt warp_size) #_ as warp_size;
  //   gpu_matrix_gather_n_underspec
  //     (gpu_matrix_subtile gm trows tcols (wid/(cols/tcols)) (wid%(cols/tcols)))
  //     warp_size;
  // };
  // forevery_map _ _ unshare_within_warp;
  // forevery_unflatten_natlt
  //   (fun (trow : natlt (rows/trows)) -> fun (tcol : natlt (cols/tcols)) ->
  //     (exists* (em : ematrix et trows tcols).
  //       (gpu_matrix_subtile gm trows tcols trow tcol) |-> Frac f em));
  // gpu_matrix_untile_underspec gm trows tcols;
}

ghost
fn forevery_factor'
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt d1 -> natlt d2 -> slprop)
  requires
    forall+ (i:natlt n). p (i / d2) (i % d2)
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
fn teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
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
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 gA eA gB eB gC bm bn bk tm tn tk wm wn fA fB nthr bid tid) **
    emp
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    // underspec not implemented anyway
    (exists* eC'. gC |-> eC')
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
      live_warp_tile #et_c #_ #(hide #nat (v rows)) #(v cols) #lC gC (v bm)
            (v bn) (v tm) (v tn) (v wm) (v wn)
            bid (tid / 32));

  ghost
  fn aux (bid : natlt nblk) (tid : natlt nthr)
  requires live_warp_tile gC bm bn tm tn wm wn (bid) (tid/warp_size)
  ensures live (
        gpu_matrix_subtile
          (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn)
            (block_tile_idx_rows (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid)
            (block_tile_idx_cols (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid))
          (wm*tm) (wn*tn)
          (warp_tile_idx_rows (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
          (warp_tile_idx_cols (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size)))
        #(1.0R /. warp_size)
  {
    rewrite live_warp_tile gC bm bn tm tn wm wn (bid) (tid/warp_size)
    as
      live (
        gpu_matrix_subtile
          (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn)
            (block_tile_idx_rows (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid)
            (block_tile_idx_cols (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid))
          (wm*tm) (wn*tn)
          (warp_tile_idx_rows (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size))
          (warp_tile_idx_cols (SZ.v bm) (SZ.v bn) (wm*tm) (wn*tn) (tid/warp_size)))
        #(1.0R /. warp_size);
  };
  forevery_map_2 #(natlt (rows/bm * (cols/bn))) #(natlt nthr) _ _ aux;

  forevery_map
    (fun (bid : natlt ((rows/bm * (cols/bn)))) -> forall+ (tid: natlt nthr).
      exists* (em: ematrix et_c (wm * tm) (wn * tn)).
        gpu_matrix_subtile
          (gpu_matrix_subtile gC (v bm) (v bn) (bid / (cols/bn)) (bid % (cols/bn)))
              (wm*tm)
              (wn*tn)
              (tid / 32 / (bn / (wn*tn)))
              (tid / 32 % (bn / (wn*tn))) |-> Frac (1.0R /. warp_size) em)
    _
    (fun (bid : natlt nblk) -> (untile_warp_tiles_shared
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn)
          (block_tile_idx_rows (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid)
          (block_tile_idx_cols (SZ.v rows) (SZ.v cols) (SZ.v bm) (SZ.v bn) bid))
        (wm*tm) (wn*tn) nthr));

  forevery_factor' ((rows/bm) * (cols/bn)) (rows/bm) (cols/bn)
    (fun (trow : natlt (rows/bm)) -> fun (tcol : natlt (cols/bn)) ->
      (exists* (em : ematrix et_c bm bn).
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) trow tcol) |-> em));
  gpu_matrix_untile_underspec gC bm bn;
  ()
}
