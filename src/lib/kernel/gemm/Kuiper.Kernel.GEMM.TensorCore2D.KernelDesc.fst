module Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

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
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.TensorCore
open Kuiper.VArray { varray, varray_pts_to, varray_pts_to_cell }
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

let bid_of_ij
  (m n : nat)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (i : natlt m)
  (j : natlt n)
  : natlt (m/bm * (n/bn))
  = (i / bm) * (n/bn) + (j / bn)

let wid_of_ij
  (m n : nat)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (i : natlt m)
  (j : natlt n)
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
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (i : natlt m)
  (j : natlt n)
  : Lemma (ensures warp_tile_i #m #n bm bn bk tm tn tk wm wn
                       nthr
                       (bid_of_ij m n bm bn i j)
                       (wid_of_ij m n bm bn tm tn wm wn i j)
                     * (wm*tm) + ((i % bm) % (wm*tm)) == i)
  = let bid = bid_of_ij m n bm bn i j in
    let wid = wid_of_ij m n bm bn tm tn wm wn i j in
    assert (j/bn < n/bn);
    silly_modulo_helper (n/bn) (i/bm) (j/bn);
    assert (bid / (n/bn) == i / bm);
    silly_div_helper (bn/(wn*tn)) ((i%bm)/(wm*tm)) ((j%bn)/(wn*tn));
    assert (wid / (bn/(wn*tn)) == (i % bm) / (wm*tm));
    calc (==) {
      warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid * (wm*tm)
        + ((i % bm) % (wm*tm));
      == {}
      ((bid / (n/bn)) * (bm / (wm*tm)) + wid / (bn/(wn*tn))) * (wm*tm)
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
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : pos { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (i : natlt m)
  (j : natlt n)
  : Lemma (ensures warp_tile_j #m #n bm bn bk tm tn tk wm wn
                       nthr
                       (bid_of_ij m n bm bn i j)
                       (wid_of_ij m n bm bn tm tn wm wn i j)
                     * (wn*tn) + ((j % bn) % (wn*tn)) == j)
  = let bid = bid_of_ij m n bm bn i j in
    let wid = wid_of_ij m n bm bn tm tn wm wn i j in
    assert (j/bn < n/bn);
    silly_modulo_helper (n/bn) (i/bm) (j/bn);
    assert (bid % (n/bn) == j / bn);
    silly_div_helper (bn/(wn*tn)) ((i%bm)/(wm*tm)) ((j%bn)/(wn*tn));
    assert (wid % (bn/(wn*tn)) == (j % bn) / (wn*tn));
    calc (==) {
      warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid * (wn*tn)
        + ((j % bn) % (wn*tn));
      == {}
      ((bid % (n/bn)) * (bn/(wn*tn)) + wid % (bn/(wn*tn))) * (wn*tn)
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
  (arr : larray a sz)
  (#f : perm)
  (m n : nat)
  (k : nat { k > 0 })
  requires
    forall+ (_ : natlt k).
      exists* v.
        pts_to_slice arr #(f /. k) m n v
  ensures
    exists* v.
      pts_to_slice arr #f m n v
{
  forevery_natlt_pop k _;
  with vv. assert pts_to_slice arr #(f /. k) m n vv;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      pts_to_slice arr #(f /. k) m n vv ** (exists* v. pts_to_slice arr #(f /. k) m n v)
    ensures
      pts_to_slice arr #(f /. k) m n vv ** pts_to_slice arr #(f /. k) m n vv
  {
    slice_pts_to_eq arr m n (f /. k) #_ #vv;
  };
  forevery_map_extra #(natlt (k-1)) (pts_to_slice arr #(f /. k) m n vv)
    (fun (_ : natlt (k-1)) -> exists* v. pts_to_slice arr #(f /. k) m n v)
    (fun (_ : natlt (k-1)) -> pts_to_slice arr #(f /. k) m n vv)
    aux;
  forevery_natlt_push k _;
  slice_gather arr m n k;
}

ghost
fn array2_share_threads
  (#et : Type)
  (#m #n : nat)
  (#l : layout2 m n)
  (gm : array2 et l)
  (#f : perm)
  (#em : chest2 et m n)
  (nblk nthr : pos)
requires
  gm |-> Frac f em
ensures
  forall+ (bid : natlt nblk) (tid : natlt nthr). gm |-> Frac (f/.(nblk*nthr)) em
{
  tensor_share_n gm (nblk*nthr);
  forevery_factor (nblk * nthr) nblk nthr _;
}

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_c lC)
  (eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /? m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /? n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
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
    pure (SZ.fits (lC.ulen)) // frame
{
  tensor_pts_to_ref gC;
  array2_share_threads gA nblk nthr;
  array2_share_threads gB nblk nthr;

  array2_tile gC bm bn;
  forevery_unfactor' nblk (m/bm) (n/bn) _;

  ghost
  fn create_warp_tiles_shared
    (#et : Type0) {| scalar et |}
    (#m #n : nat)
    (#l : layout2 m n)
    ([@@@mkey] gm : array2 et l)
    (#f : perm)
    (#em : chest2 et m n)
    (trows : nat{trows > 0 /\ trows /? m})
    (tcols : nat{tcols > 0 /\ tcols /? n})
    (nthr : nat{nthr == m/trows * (n/tcols) * warp_size})
  requires
    gm |-> Frac f em
  ensures
    forall+ (trc : natlt nthr).
      warp_tile gm trows tcols (trc/warp_size)
        |-> Frac (f /. warp_size)
      (ematrix_subtile em trows tcols
        (warp_tile_idx_rows m n trows tcols (trc/warp_size))
        (warp_tile_idx_cols m n trows tcols (trc/warp_size)))
  {
    array2_tile gm trows tcols;
    forevery_unfactor' (m/trows * (n/tcols)) (m/trows) (n/tcols) _;

    forevery_map
      (fun (trc : natlt (m/trows * (n/tcols))) ->
        array2_subtile gm trows tcols (trc/(n/tcols)) (trc%(n/tcols))
          |-> Frac f (ematrix_subtile em trows tcols (trc/(n/tcols)) (trc%(n/tcols))))
      (fun trc ->
        forall+ (_lid: natlt warp_size).
          array2_subtile gm trows tcols (trc/(n/tcols)) (trc%(n/tcols))
            |-> Frac (f /. warp_size) (ematrix_subtile em trows tcols (trc/(n/tcols)) (trc%(n/tcols))))
      fn trc { tensor_share_n _ warp_size };
    forevery_unfactor' nthr (m / trows * (n / tcols)) 32 _;
    ();
  };
  forevery_map
    (fun (trc : natlt nblk) ->
      (array2_subtile gC (SZ.v bm) (SZ.v bn) (trc/(n/bn)) (trc%(n/bn)))
        // Explicit fraction required, otherwise tactic to resolve it fails?!?!
        |-> Frac 1.0R
      (ematrix_subtile eC bm bn (trc/(n/bn)) (trc%(n/bn))))
    _
    (fun trc ->
      create_warp_tiles_shared
        (block_tile gC (SZ.v bm) (SZ.v bn) trc)
        // (array2_subtile gC (SZ.v bm) (SZ.v bn) (trc/(n/bn)) (trc%(n/bn)))
        (wm*tm)
        (wn*tn)
        nthr);

  forevery_zip_2 #(natlt nblk) #(natlt nthr)
    (fun bid -> fun tid -> gB |-> Frac (fB /. (nblk*nthr)) eB)
    (fun bid -> fun tid ->
      (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) (tid/warp_size))
        |-> Frac (precip warp_size)
      (ematrix_subtile (ematrix_subtile eC bm bn (bid/(n/bn)) (bid%(n/bn)))
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
      |-> Frac (precip warp_size)
    (ematrix_subtile (ematrix_subtile eC bm bn (bid/(n/bn)) (bid%(n/bn)))
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
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, v : has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_c lC)
  (eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size /\ nthr <= 1024})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid)
  ensures
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid) **
    emp (* frame *)
{
  (* permissions for shared memory *)
  // shmem for A tile
  gpu_live_c_shmems_share_underspec sh #(1.0R) #nthr;

  (* consolidate permissions under a single forall+ *)
  forevery_zip #(natlt nthr)
    (fun tid -> kpre1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) _;
  ()
}

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_c lC)
  (eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (fA fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
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

  (* Restore and give back ownership of shared memory arrays. *)
  gpu_live_c_shmems_gather_underspec sh #(1.0R) #nthr;
}


ghost
fn warp_tile_pts_to_eq
  (#et : Type0) {| scalar et |}
  (#m : nat)
  (#n : nat)
  (#lC : layout2 m n)
  (gC : array2 et lC)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (em1 em2 : chest2 et (wm * tm) (wn * tn))
  requires
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em1 **
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2
  ensures
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2 **
    warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2
{
  unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em1;
  unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2;
  tensor_pts_to_eq
    (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
    (precip warp_size)
    #em1 #em2;
  fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2;
  fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em2;
}

ghost
fn warp_tile_pts_to_gatherwarp
  (#et : Type0) {| scalar et |}
  (#m : nat)
  (#n : nat)
  (#lC : layout2 m n)
  (gC : array2 et lC)
  (bm : pos{bm /?+ m})
  (bn : pos{bn /?+ n})
  (tm : pos{tm /?+ bm})
  (tn : pos{tn /?+ bn})
  (wm : pos{wm * tm /?+ bm})
  (wn : pos{wn * tn /?+ bn})
  (bid : natlt ((m/bm) * (n/bn)))
  (wid : natlt (bm/(wm*tm) * (bn/(wn*tn))))
  (em : chest2 et (wm * tm) (wn * tn))
  requires
    forall+ (_ : natlt warp_size).
      warp_tile_pts_to gC bm bn tm tn wm wn bid wid em
  ensures
    warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid em
{
  forevery_map #(natlt warp_size)
    (fun _ -> warp_tile_pts_to gC bm bn tm tn wm wn bid wid em)
    (fun _ ->
      tensor_pts_to
        (warp_tile (block_tile gC bm bn bid) (wm*tm) (wn*tn) wid)
        #(precip warp_size)
        em)
    fn _ {
      unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid em;
    };
  tensor_gather_n
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
  (#m #n #k : pos)
  (#lC : layout2 m n)
  (gC : array2 et_c lC)
  (eA : chest2 et_ab m k)
  (eB : chest2 et_ab k n)
  (eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : pos{nblk == m/bm * (n/bn)})
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (bid : natlt nblk)
  (emf : natlt nthr -> chest2 et_c (wm*tm) (wn*tn))
  norewrite
  requires
    forall+ (tid : natlt nthr).
      warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) (emf tid) **
      pure (emf tid %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))))
  ensures
    forall+ (wid : natlt (nthr / warp_size)).
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf (wid * warp_size)) **
      pure (emf (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid))))
{
  forevery_factor nthr (nthr / warp_size) warp_size _;
  ghost
  fn aux (wid : natlt (nthr / warp_size))
    requires
      forall+ (i : natlt warp_size).
        warp_tile_pts_to gC bm bn tm tn wm wn bid ((wid * warp_size + i) / warp_size) (emf (wid * warp_size + i)) **
        pure (emf (wid * warp_size + i) %~
          (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)) 0)
                     (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)))))
    ensures
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf (wid * warp_size)) **
      pure (emf (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                  (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid))))
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
    //   (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)) 0)
    //              (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + i) / warp_size)))));
    forevery_map_extra
      #(natlt (warp_size - 1))
      (warp_tile_pts_to gC bm bn tm tn wm wn bid wid em0)
      (fun i -> warp_tile_pts_to gC bm bn tm tn wm wn bid ((wid * warp_size + (i+1)) / warp_size) (emf (wid * warp_size + (i+1)))
        ** pure (emf (wid * warp_size + (i+1)) %~
          (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + (i+1)) / warp_size)) 0)
                     (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid ((wid * warp_size + (i+1)) / warp_size))))))
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
  (#m #n #k : pos)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : pos{nblk == m/bm * (n/bn)})
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (eC' : chest2 et_c m n)
  (emf : natlt nblk -> natlt nthr -> chest2 et_c (wm*tm) (wn*tn))
  : Lemma (requires
            (eC' ==
              ematrix_from_tiles bm bn
                (fun br bc ->
                  ematrix_from_tiles (wm*tm) (wn*tn)
                    (fun tr tc -> emf (br * (n/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size))))
          /\ (forall (bid : natlt nblk) (wid : natlt (nthr / warp_size)).
                emf bid (wid * warp_size) %~
                  (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                            (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid)))))
          (ensures eC' %~ MS.matmul rA rB)
= let aux (i : natlt m) (j : natlt n)
    : Lemma (acc2 eC' i j %~ acc2 (MS.matmul rA rB) i j)
  =
    let bid : natlt nblk = bid_of_ij m n bm bn i j in
    let wid : natlt (nthr / warp_size) = wid_of_ij m n bm bn tm tn wm wn i j in
    let ii  : natlt (wm*tm) = (i % bm) % (wm*tm) in
    let jj  : natlt (wn*tn) = (j % bn) % (wn*tn) in
    calc (==) {
      acc2 eC' i j;
      == {}
      acc2 (ematrix_from_tiles #_ #m #n bm bn
              (fun br bc ->
                ematrix_from_tiles (wm*tm) (wn*tn)
                  (fun tr tc -> emf (br * (n/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size))))
        i j;
      == {}
      acc2 (ematrix_from_tiles #_ #bm #bn (wm*tm) (wn*tn)
                  (fun tr tc -> emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size)))
           (i % bm) (j % bn);
      == {}
      acc2 (emf bid (wid * warp_size)) ii jj;
    };
    assert (acc2 eC' i j == acc2 (emf bid (wid * warp_size)) ii jj);
    assert (acc2 eC' i j %~
      acc2
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid)))
        ii jj);

    MS.matmul_decompose_lemma rA rB (wm*tm) (wn*tn) (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid)
                                                    (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid);
    assert (acc2 eC' i j %~
      acc2
        (ematrix_subtile
          (MS.matmul rA rB)
          (wm*tm) (wn*tn)
          (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid)
          (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid))
        ii jj);

    calc (==) {
      acc2
        (ematrix_subtile
          (MS.matmul rA rB)
          (wm*tm) (wn*tn)
          (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid)
          (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid))
        ii jj;
      == {}
      acc2
        (MS.matmul rA rB)
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid * (wm*tm) + ii)
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid * (wn*tn) + jj);
      == { lem_i m n k bm bn bk tm tn tk wm wn nthr i j;
           lem_j m n k bm bn bk tm tn tk wm wn nthr i j }
      acc2 (MS.matmul rA rB) i j;
    };

    ()
  in
  Classical.forall_intro_2 aux;
  ()
#pop-options

#push-options "--z3rlimit 100 --retry 2" // the function below is pretty terribly performant
ghost
fn reconstruct_from_warp_approx
  (#et_ab #et_c : Type0)
  {| scalar et_c |}
  {| real_like et_c |}
  (#m #n #k : pos)
  (#lC : layout2 m n)
  (gC : array2 et_c lC)
  (eA : chest2 et_ab m k)
  (eB : chest2 et_ab k n)
  (eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : pos{nblk == m/bm * (n/bn)})
  (nthr : pos{nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  norewrite
  requires
    pure (SZ.fits (lC.ulen))
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))
  ensures
    exists* (eC' : chest2 et_c m n).
      gC |-> eC' ** pure (eC' %~ MS.matmul rA rB)
{
  (* Expose the existential. *)
  forevery_map_2 #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))))
    (fun bid tid ->
      exists* (em : chest2 et_c (wm*tm) (wn*tn)).
        warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) em **
        pure (em %~
          (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                     (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))))
    fn bid tid {
      unfold warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))));
    };

  (* Choice. *)
  let emf = forevery_exists_2
    (fun (bid : natlt nblk) (tid : natlt nthr) (em : chest2 et_c (wm*tm) (wn*tn)) ->
        warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) em **
        pure (em %~
          (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                     (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))));

  (* Threads within a warp are uniform, factor and gather. *)
  forevery_map #(natlt nblk)
    (fun bid -> forall+ (tid : natlt nthr).
      warp_tile_pts_to gC bm bn tm tn wm wn bid (tid / warp_size) (emf bid tid) **
      pure (emf bid tid %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))))))
    (fun bid -> forall+ (wid : natlt (nthr / warp_size)).
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)) **
      pure (emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid)))))
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
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid)))))
    (fun bid wid ->
      emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid))))
    fn bid wid {
      ();
    };
  assert pure (forall (bid : natlt nblk) (wid : natlt (nthr / warp_size)).
    emf bid (wid * warp_size) %~
      (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                 (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid))));

  (* Now drop the pures. *)
  forevery_map_2
    #(natlt nblk) #(natlt (nthr / warp_size))
    (fun bid wid ->
      warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)) **
      pure (emf bid (wid * warp_size) %~
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid wid) 0)
                   (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid wid)))))
    (fun bid wid ->
      // warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size)))
      // tensor_pts_to
      //   (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) wid)
      //   (emf bid (wid * warp_size)))
      tensor_pts_to
        (array2_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
          (wm*tm) (wn*tn)
          (wid / (bn/(wn*tn))) (wid % (bn/(wn*tn))))
        (emf bid (wid * warp_size)))
    fn bid wid {
      unfold warp_tile_pts_to_full gC bm bn tm tn wm wn bid wid (emf bid (wid * warp_size));
      rewrite each
        (warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn) wid)
      as
        array2_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
          (wm*tm) (wn*tn)
          (wid / (bn/(wn*tn))) (wid % (bn/(wn*tn)));
      ();
    };

  (* First join every block tile *)
  forevery_map
    #(natlt nblk)
    (fun bid -> forall+ (wid : natlt (nthr / warp_size)).
      tensor_pts_to
        (array2_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
          (wm*tm) (wn*tn)
          (wid / (bn/(wn*tn))) (wid % (bn/(wn*tn))))
        (emf bid (wid * warp_size)))
    (fun bid ->
      tensor_pts_to (block_tile gC (v bm) (v bn) bid)
        (ematrix_from_tiles (wm*tm)
            (wn*tn)
            (fun tr tc -> emf bid ((tr * (bn/(wn*tn)) + tc) * 32))))
    fn bid {
      forevery_factor (nthr / warp_size) (bm/(wm*tm)) (bn/(wn*tn)) _;
      assert pure (forall (tr : natlt (bm/(wm*tm))) (tc : natlt (bn/(wn*tn))).
                    (tr * (bn/(wn*tn)) + tc) / (bn/(wn*tn))  == tr);
      assert pure (forall (tr : natlt (bm/(wm*tm))) (tc : natlt (bn/(wn*tn))).
                    (tr * (bn/(wn*tn)) + tc) % (bn/(wn*tn))  == tc);
      forevery_ext_2 #(natlt (bm/(wm*tm))) #(natlt (bn/(wn*tn)))
        (fun tr tc ->
          tensor_pts_to
            (array2_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
              (wm*tm) (wn*tn)
              ((tr * (bn/(wn*tn)) + tc) / (bn/(wn*tn))) ((tr * (bn/(wn*tn)) + tc) % (bn/(wn*tn))))
            (emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size)))
        (fun tr tc ->
          tensor_pts_to
            (array2_subtile (block_tile gC (SZ.v bm) (SZ.v bn) bid)
              (wm*tm) (wn*tn)
              tr tc)
            (emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size)))
        ;
      array2_untile' (block_tile gC (SZ.v bm) (SZ.v bn) bid) (wm*tm) (wn*tn)
        (fun tr tc ->
          emf bid ((tr * (bn/(wn*tn)) + tc) * warp_size));
    };

  (* Now that we have each block tile, again shuffle them and join. *)
  assert (forall+ (bid : natlt nblk).
      tensor_pts_to (block_tile gC (v bm) (v bn) bid)
        (ematrix_from_tiles (wm*tm)
            (wn*tn)
            (fun tr tc -> emf bid ((tr * (bn/(wn*tn)) + tc) * 32))));
  forevery_factor nblk (m/bm) (n/bn) _;
  forevery_map_2 #(natlt (m/bm)) #(natlt (n/bn))
    (fun br bc ->
      tensor_pts_to (block_tile gC (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc))
        (ematrix_from_tiles (wm*tm) (wn*tn)
            (fun tr tc -> emf (br * (n/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * 32))))
    (fun br bc ->
      tensor_pts_to
        (array2_subtile gC (SZ.v bm) (SZ.v bn) br bc)
        (ematrix_from_tiles (wm*tm) (wn*tn)
            (fun tr tc -> emf (br * (n/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * 32))))
    fn br bc {
      rewrite each
        block_tile gC (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc)
      as
        array2_subtile gC (SZ.v bm) (SZ.v bn)
          (block_tile_idx_rows m n (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc))
          (block_tile_idx_cols m n (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc));
      assert pure
        (block_tile_idx_rows m n (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc)
        ==
        (br * (n/bn) + bc) / (n/bn)
      );
      assert pure ((br * (n/bn) + bc) / (n/bn) == br);
      rewrite each
        block_tile_idx_rows m n (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc)
      as
        br;
      assert pure (
        block_tile_idx_cols m n (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc)
        ==
        (br * (n/bn) + bc) % (n/bn)
      );
      assert pure ((br * (n/bn) + bc) % (n/bn) == bc);
      rewrite each
        block_tile_idx_cols m n (SZ.v bm) (SZ.v bn) (br * (n/bn) + bc)
      as
        bc;

      // These are needed to change the arguments to the implicit layout of gC...
      rewrite each ((br * (n/bn) + bc) / (n/bn)) as br;
      rewrite each ((br * (n/bn) + bc) % (n/bn)) as bc;

      assert
        array2_subtile gC (SZ.v bm) (SZ.v bn) br bc |->
          ematrix_from_tiles (wm*tm) (wn*tn)
            (fun tr tc -> emf (br * (n/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size));

      ();
    };

  array2_untile' gC bm bn
    (fun br bc ->
      ematrix_from_tiles (wm*tm) (wn*tn)
        (fun tr tc -> emf (br * (n/bn) + bc) ((tr * (bn/(wn*tn)) + tc) * warp_size)));

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
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_c lC)
  (eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bk /?+ k))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (fA fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid) **
    pure (SZ.fits (lC.ulen)) // frame
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : chest2 et_c m n).
      gC |-> eC' ** pure (eC' %~ MS.matmul rA rB))
{
  forevery_unfactor' (m/bm * (n/bn) * nthr) nblk nthr _;
  forevery_unzip _ _;
  forevery_unzip _ _;

  tensor_gather_n gA (m/bm * (n/bn) * nthr);
  tensor_gather_n gB (m/bm * (n/bn) * nthr);

  (* Done with gA and gB. The tricky bit is getting back gC and
  proving it approximates the matmul. *)

  assert forall+ (x : natlt (m / bm * (n / bn) * nthr)).
    warp_tile_approximates gC bm bn tm tn wm wn (x / nthr) (x % nthr / warp_size)
      (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i bm bn bk tm tn tk wm wn nthr (x / nthr) (x % nthr / warp_size)) 0)
                 (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j bm bn bk tm tn tk wm wn nthr (x / nthr) (x % nthr / warp_size))));

  forevery_factor'
    (m/bm * (n/bn) * nthr)
    (m/bm * (n/bn))
    nthr
    (fun (bid : natlt (m/bm * (n/bn))) (tid : natlt nthr) ->
      warp_tile_approximates gC bm bn tm tn wm wn bid (tid / warp_size)
        (MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
                  (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))));

  reconstruct_from_warp_approx
    gC eA eB eC bm bn bk tm tn tk wm wn (m/bm * (n/bn)) nthr rA rB rC;

  ();
}

