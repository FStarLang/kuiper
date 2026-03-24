module Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module B = Kuiper.Barrier
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT

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
