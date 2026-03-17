module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell, }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module B = Kuiper.Barrier
module M = Kuiper.Matrix
module MU = Kuiper.Poly.GEMM.Util
module R = Kuiper.Matrix.Reprs
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade

open Kuiper.Poly.GEMM.Copy { live_cell }

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc
  (et:Type0) {| sized et |}
  (bm bn bk: szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  : list shmem_desc = [
  SHArray et (bm *^ bk);
  SHArray et (bk *^ bn);
]

let constraint_test
  (#bm #bn #bk : szp)
  (tm : szp{tm /?+ bm})
  // because of how the original populates shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  =
  assert (bm == bn);
  assert ((bm/tm * bn) == bm * bk /\ (bm/tm * bm) == bn * bk);
  assert ((bm/tm * bm) == bm * bk /\ (bm/tm * bm) == bn * bk);
  assert ((bm/tm * bm) == bm * bk /\ bm * bk == bn * bk);
  assert ((bm/tm * bm) == bm * bk /\ bm == bn);
  assert (SZ.v bm == tm * bk /\ bm == bn);
  // ^ A simpler constraint
  ()

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  // because of how the original populates shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  // as many blocks as output tiles (i.e. tiles in gC)
  (bid : natlt (mrows * mcols))
  //  so elements in a tile divided by tm
  // each thread in a block computes tm many elements in M dimension,
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn))) eA **
  gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn))) eB **
  forall+ (i : natlt tm).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
      // Each thread computes tm many results in a subcolumn of C.
      // bn threads next to each other compute an innertilerow,
      // sharing the row indices
      ((tid / bn * tm) + i)
      // and not sharing the column indices
      (tid % bn)
      (macc
        (ematrix_subtile eC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
        ((tid / bn * tm) + i)
        (tid % bn))

unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn))) eA **
  gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn))) eB **
  forall+ (i : natlt tm).
    exists* (v : et).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
      ((tid / bn * tm) + i)
      (tid % bn)
      v **
    pure (v %~ MU.real_gemm_single comb_r eA eB eC
            (mrow * bm + (tid / bn * tm) + i)
            (mcol * bn + (tid % bn)))

(* The barrier flip-flops between an initial state
where every threads shares all of the two arrays, and
a second state where every thread owns a single cell
in each array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)
(* Per-cell barrier contract following SHMem's pattern:
   threads flip-flop between shared fractional ownership (even steps)
   and exclusive single-cell ownership (odd steps).

   Even steps (rin): each thread gives back fractional read ownership
   Odd steps (rin): each thread gives back single-cell write ownership
     with *specific* content (the correct subtile) so the proof goes through.
   Even steps (rout): each thread receives single-cell write ownership
   Odd steps (rout): each thread receives fractional read ownership
     with *specific* content (the correct subtile). *)
let barrier_p
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols * bn))
  (bid : natlt (mrows * mcols))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  : B.barrier_side (bm/tm * bn) =
  fun it tid ->
    if it >= 2 * mshared then
      emp
    else if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x)
    else
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow (it / 2)) (tid/bk) (tid%bk)) **
      gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn (it / 2) mcol) (tid/bn) (tid%bn))

let barrier_q
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols * bn))
  (bid : natlt (mrows * mcols))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  : B.barrier_side (bm/tm * bn) =
  fun it tid ->
    if it >= 2 * mshared then
      emp
    else if even it then
      live_cell m1 (tid/bk) (tid%bk) ** live_cell m2 (tid/bn) (tid%bn)
    else
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it / 2)) **
      m2 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eB bk bn (it / 2) mcol)

let barrier_contract
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols * bn))
  (bid : natlt (mrows * mcols))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2) : B.contract (bm/tm * bn) = {
    rin  = barrier_p tm eA eB bid m1 m2;
    rout = barrier_q tm eA eB bid m1 m2;
  }

#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols * bn))
  (bid : natlt (mrows * mcols))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (#_ : squash (SZ.fits (mlayout_size l1)))
  (#_ : squash (SZ.fits (mlayout_size l2)))
  (it : nat)
  requires
    forall+ (tid : natlt (bm/tm * bn)).
      barrier_p tm eA eB bid m1 m2 it tid
  ensures
    forall+ (tid : natlt (bm/tm * bn)).
      barrier_q tm eA eB bid m1 m2 it tid
{
  if (it >= 2 * mshared) {
    forevery_map
      (fun (tid : natlt (bm/tm * bn)) -> barrier_p tm eA eB bid m1 m2 it tid)
      (fun (tid : natlt (bm/tm * bn)) -> barrier_q tm eA eB bid m1 m2 it tid)
      fn tid {
        rewrite barrier_p tm eA eB bid m1 m2 it tid as emp;
        rewrite emp as barrier_q tm eA eB bid m1 m2 it tid;
      };
  } else {
    let ev = even it;
    if ev {
      assert pure (it < 2 * mshared);
      assert pure (even it);
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> barrier_p tm eA eB bid m1 m2 it tid)
        (fun (tid : natlt (bm/tm * bn)) ->
          (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
          (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x))
        fn tid {
          rewrite barrier_p tm eA eB bid m1 m2 it tid
               as (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
                  (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x);
        };
      forevery_unzip _ _;
      M.gpu_matrix_gather_n_underspec m1 (bm/tm * bn);
      with em1. assert m1 |-> em1;
      M.gpu_matrix_explode m1;
      forevery_unfactor' (bm/tm * bn) bm bk
        (fun r c -> gpu_matrix_pts_to_cell m1 r c (macc em1 r c));
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc em1 (tid/bk) (tid%bk)))
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m1 (tid/bk) (tid%bk))
        fn tid { fold (live_cell m1 (tid/bk) (tid%bk)) };
      M.gpu_matrix_gather_n_underspec m2 (bm/tm * bn);
      with em2. assert m2 |-> em2;
      M.gpu_matrix_explode m2;
      forevery_unfactor' (bm/tm * bn) bk bn
        (fun r c -> gpu_matrix_pts_to_cell m2 r c (macc em2 r c));
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc em2 (tid/bn) (tid%bn)))
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m2 (tid/bn) (tid%bn))
        fn tid { fold (live_cell m2 (tid/bn) (tid%bn)) };
      forevery_zip
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m1 (tid/bk) (tid%bk)) _;
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m1 (tid/bk) (tid%bk) ** live_cell m2 (tid/bn) (tid%bn))
        (fun (tid : natlt (bm/tm * bn)) -> barrier_q tm eA eB bid m1 m2 it tid)
        fn tid {
          rewrite live_cell m1 (tid/bk) (tid%bk) ** live_cell m2 (tid/bn) (tid%bn)
               as barrier_q tm eA eB bid m1 m2 it tid;
        };
    } else {
      assert pure (it < 2 * mshared);
      assert pure (odd it);
      let mrow = bid / mcols;
      let mcol = bid % mcols;
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> barrier_p tm eA eB bid m1 m2 it tid)
        (fun (tid : natlt (bm/tm * bn)) ->
          gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow (it/2)) (tid/bk) (tid%bk)) **
          gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn (it/2) mcol) (tid/bn) (tid%bn)))
        fn tid {
          rewrite barrier_p tm eA eB bid m1 m2 it tid
               as gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow (it/2)) (tid/bk) (tid%bk)) **
                  gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn (it/2) mcol) (tid/bn) (tid%bn));
        };
      forevery_unzip _ _;
      forevery_factor' (bm/tm * bn) bm bk
        (fun r c -> gpu_matrix_pts_to_cell m1 r c (macc (ematrix_subtile eA bm bk mrow (it/2)) r c));
      M.gpu_matrix_implode m1;
      M.gpu_matrix_share_n m1 (bm/tm * bn);
      forevery_factor' (bm/tm * bn) bk bn
        (fun r c -> gpu_matrix_pts_to_cell m2 r c (macc (ematrix_subtile eB bk bn (it/2) mcol) r c));
      M.gpu_matrix_implode m2;
      M.gpu_matrix_share_n m2 (bm/tm * bn);
      forevery_zip
        (fun (_ : natlt (bm/tm * bn)) -> m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it/2))) _;
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) ->
          m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it/2)) **
          m2 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eB bk bn (it/2) mcol))
        (fun (tid : natlt (bm/tm * bn)) -> barrier_q tm eA eB bid m1 m2 it tid)
        fn tid {
          rewrite
            m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it/2)) **
            m2 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eB bk bn (it/2) mcol)
          as
            barrier_q tm eA eB bid m1 m2 it tid;
        };
    }
  }
}
#pop-options

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn) // shmem layouts
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  kpre1 comb tm gA gB gC eA eB eC fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x)

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn) // shmem layouts
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
  : slprop
  =
  kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x)

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : erased nat)
  (tm : szp{tm /?+ bm})
  // every thread loads a single element for either matrix,
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#slA : full_mlayout bm bk) {| clayout slA |}
  (#slB : full_mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (tid : szlt (bm/tm * bn))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    live_cell sA (tid/bk) (tid%bk) **
    live_cell sB (tid/bn) (tid%bn)
  ensures
    gpu_matrix_pts_to_cell sA (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow mk) (tid/bk) (tid%bk)) **
    gpu_matrix_pts_to_cell sB (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn mk mcol) (tid/bn) (tid%bn))
{
  unfold live_cell sA (tid/bk) (tid%bk);
  let tA = gpu_matrix_extract_tile_ro' gA (SZ.v bm) (SZ.v bk) (SZ.v mrow) (SZ.v mk);
  let va = M.gpu_matrix_read tA (tid /^ bk) (tid %^ bk);
  M.gpu_matrix_write_cell sA (tid /^ bk) (tid %^ bk) va;
  Trade.elim_trade _ _;
  rewrite each va as (macc (ematrix_subtile eA bm bk mrow mk) (tid/bk) (tid%bk));

  unfold live_cell sB (tid/bn) (tid%bn);
  let tB = gpu_matrix_extract_tile_ro' gB (SZ.v bk) (SZ.v bn) (SZ.v mk) (SZ.v mcol);
  let vb = M.gpu_matrix_read tB (tid /^ bn) (tid %^ bn);
  M.gpu_matrix_write_cell sB (tid /^ bn) (tid %^ bn) vb;
  Trade.elim_trade _ _;
  rewrite each vb as (macc (ematrix_subtile eB bk bn mk mcol) (tid/bn) (tid%bn));
}

inline_for_extraction noextract
fn subproducts1d
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk: szp)
  (tm : szp{tm /?+ bm})
  (rch1d : larray et tm) (* register cache, 1d *)
  (#resvs : erased (seq et))
  (#l1 : full_mlayout bm bk) {| clayout l1 |}
  (#l2 : full_mlayout bk bn) {| clayout l2 |}
  (gA : gpu_matrix et l1)
  (gB : gpu_matrix et l2)
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt bn)
  preserves
    gpu **
    gA |-> Frac f eA **
    gB |-> Frac f eB
  requires
    rch1d |-> resvs
  ensures
    exists* resvs'.
      rch1d |-> resvs'
{
  let mut dotIdx : sz = 0sz;
  while (!dotIdx <^ bk)
    invariant live dotIdx ** pure (!dotIdx <= bk)
    invariant live rch1d
  {
    let tmpB = M.gpu_matrix_read gB !dotIdx bcol;
    let mut resIdx = 0sz;
    while (SZ.(!resIdx <^ tm))
      invariant live resIdx ** pure (!resIdx <= tm)
      invariant live rch1d
    {
      let va = M.gpu_matrix_read gA (arow *^ tm +^ !resIdx) !dotIdx;

      open Pulse.Lib.Array;
      Pulse.Lib.Array.pts_to_len rch1d;
      let sum0 = rch1d.(!resIdx);
      let sum1 = sum0 `add` (va `mul` tmpB);
      rch1d.(!resIdx) <- sum1;
      resIdx := !resIdx +^ 1sz;
    };
    dotIdx := !dotIdx +^ 1sz;
  }
}

// even 20 isn't evenough for the checking from the terminal
//  (but enough for the vs code extension)
#push-options "--z3rlimit 50"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA : perm)
  (#fB : perm)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (mrows * mcols))
  (tid : szlt (bm/tm * bn))
  ()
  norewrite
  requires
    gpu **
    kpre comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (barrier_contract tm eA eB bid (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (barrier_contract tm eA eB bid (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state (2 * mshared)
{
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  M.gpu_matrix_abs' slA sarA;
  let sA = M.from_array slA sarA;
  rewrite each M.from_array slA sarA as sA;

  M.gpu_matrix_abs' slB sarB;
  let sB = M.from_array slB sarB;
  rewrite each M.from_array slB sarB as sB;

  let mrow = bid /^ mcols;
  let mcol = bid %^ mcols;
  let threadRow = tid /^ bn;
  let threadCol = tid %^ bn;

  (* thread-local result cache *)
  let mut cache1d : Pulse.Lib.Array.array et = [| zero #et #_ ; tm |];

  let mut bkIdx  : sz = 0sz;
  while (!bkIdx <^ mshared)
    invariant
      live bkIdx ** pure (!bkIdx <= mshared) **
      B.barrier_state (2 * !bkIdx)
    invariant
      live cache1d
    invariant
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x)
  {
    even_2x !bkIdx;
    rewrite (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
            (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x)
         as barrier_p tm eA eB bid sA sB (2 * !bkIdx) tid;
    rewrite barrier_p tm eA eB bid sA sB (2 * !bkIdx) tid
         as (barrier_contract tm eA eB bid sA sB).rin (2 * !bkIdx) tid;

    B.barrier_wait ();

    rewrite (barrier_contract tm eA eB bid sA sB).rout (2 * !bkIdx) tid
         as barrier_q tm eA eB bid sA sB (2 * !bkIdx) tid;
    rewrite barrier_q tm eA eB bid sA sB (2 * !bkIdx) tid
         as live_cell sA (tid/bk) (tid%bk) ** live_cell sB (tid/bn) (tid%bn);

    (* At this point we exclusively own a cell of the SHMEM
       cache. Populate it with specific content. *)
    populate_shmem tm sA sB gA gB mrow !bkIdx mcol tid;

    odd_2x1 !bkIdx;
    assert pure (odd (2 * !bkIdx + 1));
    rewrite gpu_matrix_pts_to_cell sA (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow !bkIdx) (tid/bk) (tid%bk)) **
            gpu_matrix_pts_to_cell sB (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn !bkIdx mcol) (tid/bn) (tid%bn))
         as barrier_p tm eA eB bid sA sB (2 * !bkIdx + 1) tid;
    rewrite barrier_p tm eA eB bid sA sB (2 * !bkIdx + 1) tid
         as (barrier_contract tm eA eB bid sA sB).rin (2 * !bkIdx + 1) tid;

    B.barrier_wait ();
    even_2x (!bkIdx + 1);
    let vbkIdx = !bkIdx;
    assert (pure (2 * (vbkIdx + 1) == 2 * vbkIdx + 1 + 1));
    assert (pure (even (2 * vbkIdx + 2)));
    assert pure ((2 * vbkIdx + 1) < 2 * mshared);
    assert pure ((2 * vbkIdx + 1) / 2 == vbkIdx);
    rewrite (barrier_contract tm eA eB bid sA sB).rout (2 * vbkIdx + 1) tid
         as barrier_q tm eA eB bid sA sB (2 * vbkIdx + 1) tid;
    rewrite barrier_q tm eA eB bid sA sB (2 * vbkIdx + 1) tid
    as
      sA |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow vbkIdx) **
      sB |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eB bk bn vbkIdx mcol);

    subproducts1d tm cache1d sA sB threadRow threadCol;

    (* Move to next tile *)
    bkIdx := !bkIdx +^ 1sz;
  };

  // Weaken the ownership of C (introduce existential)
  ghost
  fn aux (i: natlt tm)
  norewrite
  requires
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
      ((tid / bn * tm) + i)
      (tid % bn)
      (macc
        (ematrix_subtile eC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
        ((tid / bn * tm) + i)
        (tid % bn))
  ensures
    exists* v.
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
        ((tid / bn * tm) + i)
        (tid % bn)
        v
  {
    ()
  };
  forevery_map _ _ aux;

  (* Write all the accumulated dotproducts. *)
  let mut resIdx : sz = 0sz;
  while (!resIdx <^ tm)
    invariant live resIdx ** pure (!resIdx <= tm)
    invariant live cache1d
  {
    Pulse.Lib.Array.pts_to_len cache1d;
    forevery_extract #(natlt tm) (SZ.v !resIdx) _;

    let tC = gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (SZ.v mrow) (SZ.v mcol);
    assert rewrites_to tC
      (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (SZ.v (bid /^ mcols)) (SZ.v (bid %^ mcols)));
    assert rewrites_to threadRow (tid /^ bn);
    assert rewrites_to threadCol (tid %^ bn);

    open Pulse.Lib.Array;
    Pulse.Lib.Array.pts_to_len cache1d;
    let v0 = M.gpu_matrix_read_cell tC (threadRow *^ tm +^ !resIdx) threadCol;
    let v1 = cache1d.(!resIdx);
    let v' = comb v0 v1;
    M.gpu_matrix_write_cell tC (threadRow *^ tm +^ !resIdx) threadCol v';

    resIdx := !resIdx +^ 1sz;
    Pulse.Lib.Trade.elim_trade _ _;
    ()
  };

  // Establish approximation spec on each cell (assumed)
  ghost
  fn aux2 (i: natlt tm)
  norewrite
  requires
    exists* v.
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
        ((tid / bn * tm) + i)
        (tid % bn)
        v
  ensures
    exists* (v : et).
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
        ((tid / bn * tm) + i)
        (tid % bn)
        v **
      pure (v %~ MU.real_gemm_single comb_r eA eB eC
              (bid / mcols * bm + (tid / bn * tm) + i)
              (bid % mcols * bn + (tid % bn)))
  {
    with v. assert (gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
        ((tid / bn * tm) + i) (tid % bn) v);
    assume pure (v %~ MU.real_gemm_single comb_r eA eB eC
            (bid / mcols * bm + (tid / bn * tm) + i)
            (bid % mcols * bn + (tid % bn)));
  };
  forevery_map _ _ aux2;

  M.gpu_matrix_concr sA; rewrite each M.core sA as sarA;
  M.gpu_matrix_concr sB; rewrite each M.core sB as sarB;

  rewrite each sA as M.from_array slA sarA;
  rewrite each sB as M.from_array slB sarB;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  ();
}
#pop-options


#push-options "--z3rlimit 80"
ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  let n_threads = (mrows * mcols) * (bm/tm * bn);

  (* Step 1: Share gA/gB, explode gC *)
  M.gpu_matrix_share_n gA n_threads;
  M.gpu_matrix_share_n gB n_threads;
  gpu_matrix_explode_tiled gC (SZ.v bm) (SZ.v bn);
  forevery_rw_size4
    ((mrows * bm) / bm) mrows
    ((mcols * bn) / bn) mcols
    (SZ.v bm) bm
    (SZ.v bn) bn;

  (* Step 2: Rearrange inner (r,c) → (tid,i) per tile *)
  forevery_map_2
    (fun (tr : natlt mrows) (tc : natlt mcols) ->
      forall+ (r : natlt bm) (c : natlt bn).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc) r c
          (macc eC (tr * bm + r) (tc * bn + c)))
    (fun (tr : natlt mrows) (tc : natlt mcols) ->
      forall+ (tid : natlt (bm/tm * bn)) (i : natlt tm).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC (tr * bm + ((tid / bn * tm) + i)) (tc * bn + (tid % bn))))
    fn tr tc {
      forevery_factor bm (bm/tm) tm
        (fun (r : natlt bm) ->
          forall+ (c : natlt bn).
            gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc) r c
              (macc eC (tr * bm + r) (tc * bn + c)));
      forevery_mid_flip
        (fun (threadRow : natlt (bm/tm)) (i : natlt tm) (c : natlt bn) ->
          gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
            (threadRow * tm + i) c
            (macc eC (tr * bm + (threadRow * tm + i)) (tc * bn + c)));
      forevery_unfactor' (bm/tm * bn) (bm/tm) bn
        (fun (threadRow : natlt (bm/tm)) (c : natlt bn) ->
          forall+ (i : natlt tm).
            gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
              (threadRow * tm + i) c
              (macc eC (tr * bm + (threadRow * tm + i)) (tc * bn + c)));
    };

  (* Step 3: Collapse (tr,tc) → bid *)
  forevery_unfactor' (mrows * mcols) mrows mcols
    (fun (tr : natlt mrows) (tc : natlt mcols) ->
      forall+ (tid : natlt (bm/tm * bn)) (i : natlt tm).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC (tr * bm + ((tid / bn * tm) + i)) (tc * bn + (tid % bn))));

  (* Step 4: Factor gA/gB to 2D *)
  forevery_factor n_threads (mrows * mcols) (bm/tm * bn)
    (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (mrows * mcols) (bm/tm * bn)
    (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  (* Step 5: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      forall+ (i : natlt tm).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                   ((bid % mcols) * bn + (tid % bn))));

  (* Step 6: Bridge to SizeT and match kpre1 *)
  forevery_rw_size2
    (mrows * mcols) (mrows *^ mcols)
    (bm/tm * bn) (bm /^ tm *^ bn);
  forevery_map_2
    (fun (bid : natlt (mrows *^ mcols)) (tid : natlt (bm /^ tm *^ bn)) ->
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (i : natlt tm).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                   ((bid % mcols) * bn + (tid % bn))))
    (fun (bid : natlt (mrows *^ mcols)) (tid : natlt (bm /^ tm *^ bn)) ->
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
    fn bid tid {
      forevery_map
        (fun (i : natlt tm) ->
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn)
            (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                     ((bid % mcols) * bn + (tid % bn))))
        (fun (i : natlt tm) ->
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn)
            (macc (ematrix_subtile eC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  ((tid / bn * tm) + i) (tid % bn)))
        fn i {
          rewrite each
            (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                     ((bid % mcols) * bn + (tid % bn)))
          as
            (macc (ematrix_subtile eC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  ((tid / bn * tm) + i) (tid % bn));
        };
    };
  ();
}
#pop-options

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows *^ mcols))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  // Bridge from SizeT-based to nat-based size
  forevery_rw_size (bm/^tm *^ bn) (bm/tm * bn);

  // Unfold live_c_shmems into individual shmem buffers
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_cons (snd sh) #1.0R;
  unfold_live_c_shmems_nil (snd (snd sh)) #1.0R;

  // Share each buffer across all threads
  gpu_live_c_shmem_share_underspec (fst sh) #1.0R #(bm/tm * bn);
  gpu_live_c_shmem_share_underspec (fst (snd sh)) #1.0R #(bm/tm * bn);

  // Unfold live_c_shmem into manual form (exists* x. arr |-> Frac ...)
  forevery_map
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmem (fst sh) #(1.0R /. (bm/tm * bn)))
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x)
    fn _ { unfold_live_c_shmem (fst sh) #(1.0R /. (bm/tm * bn)) };
  forevery_map
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmem (fst (snd sh)) #(1.0R /. (bm/tm * bn)))
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x)
    fn _ { unfold_live_c_shmem (fst (snd sh)) #(1.0R /. (bm/tm * bn)) };

  // Zip the two shmem forall+ together
  forevery_zip
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x);

  // Zip with kpre1 to form kpre
  forevery_zip
    (fun (tid : natlt (bm/tm * bn)) -> kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
    (fun (_ : natlt (bm/tm * bn)) ->
      (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x));

  // Bridge back from nat-based to SizeT-based size
  forevery_rw_size (bm/tm * bn) (bm/^tm *^ bn);
}

#push-options "--z3rlimit 120"
ghost
fn block_teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows *^ mcols))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
{
  // Bridge from SizeT-based to nat-based size
  forevery_rw_size (bm/^tm *^ bn) (bm/tm * bn);

  // Split kpost (= kpost1 ** shmemA ** shmemB) into three components
  forevery_unzip3
    (fun (tid : natlt (bm/tm * bn)) -> kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x);

  // Fold each shmem buffer into live_c_shmem, then gather
  forevery_map
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmem (fst sh) #(1.0R /. (bm/tm * bn)))
    fn _ { fold_live_c_shmem (fst sh) #(1.0R /. (bm/tm * bn)) };
  gpu_live_c_shmem_gather_underspec (fst sh) #1.0R #(bm/tm * bn);

  forevery_map
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmem (fst (snd sh)) #(1.0R /. (bm/tm * bn)))
    fn _ { fold_live_c_shmem (fst (snd sh)) #(1.0R /. (bm/tm * bn)) };
  gpu_live_c_shmem_gather_underspec (fst (snd sh)) #1.0R #(bm/tm * bn);

  // Combine into live_c_shmems
  fold_live_c_shmems_nil (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;

  // Bridge back from nat-based to SizeT-based size
  forevery_rw_size (bm/tm * bn) (bm/^tm *^ bn);
}
#pop-options

#push-options "--z3rlimit 160"
ghost
fn teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  ()
  norewrite
  requires
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et _ _).
      gC |-> eC' **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  let n_threads = (mrows * mcols) * (bm/tm * bn);

  (* Step 1: Bridge from SizeT to nat *)
  forevery_rw_size2
    (mrows *^ mcols) (mrows * mcols)
    (bm /^ tm *^ bn) (bm/tm * bn);

  (* Step 2: Unfold kpost1 *)
  forevery_ext_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (mrow * bm + (tid / bn * tm) + i)
                  (mcol * bn + (tid % bn))));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (mrow * bm + (tid / bn * tm) + i)
                  (mcol * bn + (tid % bn))));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      forall+ (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (mrow * bm + (tid / bn * tm) + i)
                  (mcol * bn + (tid % bn))));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' n_threads (mrows * mcols) (bm/tm * bn)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gA |-> Frac (fA /. n_threads) eA);
  M.gpu_matrix_gather_n gA n_threads;

  forevery_unfactor' n_threads (mrows * mcols) (bm/tm * bn)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gB |-> Frac (fB /. n_threads) eB);
  M.gpu_matrix_gather_n gB n_threads;

  (* Step 6: Rearrange (bid, tid, i) → (bid, flatid) *)
  forevery_map
    (fun (bid : natlt (mrows * mcols)) ->
      forall+ (tid : natlt (bm/tm * bn)) (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (bid / mcols * bm + (tid / bn * tm) + i)
                  (bid % mcols * bn + (tid % bn))))
    (fun (bid : natlt (mrows * mcols)) ->
      forall+ (flatid : natlt (bm * bn)).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            (flatid / bn) (flatid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (bid / mcols * bm + flatid / bn)
                  (bid % mcols * bn + flatid % bn)))
    fn bid {
      forevery_factor' (bm/tm * bn) (bm/tm) bn
        (fun (threadRow : natlt (bm/tm)) (c : natlt bn) ->
          forall+ (i : natlt tm).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                (threadRow * tm + i) c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + (threadRow * tm) + i)
                      (bid % mcols * bn + c)));
      forevery_mid_flip
        (fun (threadRow : natlt (bm/tm)) (c : natlt bn) (i : natlt tm) ->
          exists* (v : et).
            gpu_matrix_pts_to_cell
              (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
              (threadRow * tm + i) c v **
            pure (v %~ MU.real_gemm_single comb_r eA eB eC
                    (bid / mcols * bm + (threadRow * tm) + i)
                    (bid % mcols * bn + c)));
      // Re-associate addition: (a + b) + c → a + (b + c)
      forevery_map_2
        (fun (threadRow : natlt (bm/tm)) (i : natlt tm) ->
          forall+ (c : natlt bn).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                (threadRow * tm + i) c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + (threadRow * tm) + i)
                      (bid % mcols * bn + c)))
        (fun (threadRow : natlt (bm/tm)) (i : natlt tm) ->
          forall+ (c : natlt bn).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                (threadRow * tm + i) c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + (threadRow * tm + i))
                      (bid % mcols * bn + c)))
        fn threadRow i {
          assert pure (bid / mcols * bm + (threadRow * tm) + i == bid / mcols * bm + (threadRow * tm + i));
          forevery_map
            (fun (c : natlt bn) ->
              exists* (v : et).
                gpu_matrix_pts_to_cell
                  (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  (threadRow * tm + i) c v **
                pure (v %~ MU.real_gemm_single comb_r eA eB eC
                        (bid / mcols * bm + (threadRow * tm) + i)
                        (bid % mcols * bn + c)))
            (fun (c : natlt bn) ->
              exists* (v : et).
                gpu_matrix_pts_to_cell
                  (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  (threadRow * tm + i) c v **
                pure (v %~ MU.real_gemm_single comb_r eA eB eC
                        (bid / mcols * bm + (threadRow * tm + i))
                        (bid % mcols * bn + c)))
            fn c {
              ();
            };
        };
      forevery_unfactor bm (bm/tm) tm
        (fun (r : natlt bm) ->
          forall+ (c : natlt bn).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                r c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + r)
                      (bid % mcols * bn + c)));
      forevery_unfactor' (bm * bn) bm bn
        (fun (r : natlt bm) (c : natlt bn) ->
          exists* (v : et).
            gpu_matrix_pts_to_cell
              (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
              r c v **
            pure (v %~ MU.real_gemm_single comb_r eA eB eC
                    (bid / mcols * bm + r)
                    (bid % mcols * bn + c)));
    };

  (* Step 7: Collect cells back into matrix *)
  let _vf = gpu_matrix_collect_approx_tiled gC (SZ.v bm) (SZ.v bn)
    mrows mcols
    (fun (row : natlt (mrows * bm)) (col : natlt (mcols * bn)) (v : et) ->
      v %~ MU.real_gemm_single comb_r eA eB eC row col);

  (* Step 8: Prove ematrix_approximates *)
  with eC'. assert (gC |-> eC');

  assert pure (forall (row:natlt (mrows * bm)) (col:natlt (mcols * bn)).
    macc eC' row col %~ MU.real_gemm_single comb_r eA eB eC row col);

  assert pure (forall (row:natlt (mrows * bm)) (col:natlt (mcols * bn)).
    macc eC' row col %~ macc (MU.real_mmcomb comb_r eC eA eB) row col);

  assert pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB));
  ();
}
#pop-options

#push-options "--z3rlimit 80"
let kpre_block_sendable
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (_: squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
: is_send_across block_of (kpre comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid)
= magic()

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (_: squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
: is_send_across block_of (kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB sh bid tid)
= magic()
#pop-options

#push-options "--z3rlimit 80"

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt (bm/tm * bn)).
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
= magic()

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt (bm/tm * bn)).
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
= magic()
#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit_factor 10 --z3refresh"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (_ : squash (mrows * mcols <= max_blocks
               /\ (bm/tm * bn) <= max_threads))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et _ _).
          gC |-> eC' **
          pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB))))
= {
  nblk = mrows *^ mcols; //SZ.uint_to_t (SZ.v mrows * SZ.v mcols);
  nthr = (bm /^ tm *^ bn);

  shmems_desc = shmems_desc et bm bn bk;

  barrier_contract = (fun bid ptrs -> barrier_contract tm eA eB bid (M.from_array slA (fst ptrs)) (M.from_array slB (fst (snd ptrs))));
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun bid ptrs -> barrier_p_to_q_transform tm eA eB bid (M.from_array slA (fst ptrs)) (M.from_array slB (fst (snd ptrs))));

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpre1  comb tm gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid);
  setup      = setup    comb tm gA gB gC #eA #eB #eC;
  teardown   = teardown comb comb_r tm gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    =  block_setup    comb tm slA slB gA gB gC #eA #eB #eC;
  block_teardown = block_teardown comb comb_r tm slA slB gA gB gC #eA #eB #eC;

  kpre      = kpre  comb tm slA slB gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB;

  f = kf comb comb_r tm slA slB gA gB gC #fA #fB #eA #eB;

  block_pre_sendable = block_pre_gpu_sendable comb tm gA gB gC eA eB eC fA fB;
  block_post_sendable = block_post_gpu_sendable comb comb_r tm gA gB gC eA eB eC fA fB;
  kpre_sendable = kpre_block_sendable comb tm slA slB gA gB gC eA eB eC fA fB;
  kpost_sendable = kpost_block_sendable comb comb_r tm slA slB gA gB gC eA eB eC fA fB
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  launch_sync (mk_kernel comb comb_r tm (R.row_major _ _) (R.row_major _ _) gA #fA gB #fB gC #eA #eB #eC ());
}
