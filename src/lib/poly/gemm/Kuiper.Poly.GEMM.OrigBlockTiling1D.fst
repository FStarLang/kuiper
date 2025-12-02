module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.EMatrix
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell, }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module B = Kuiper.Barrier
module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module R = Kuiper.Matrix.Reprs
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade

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
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  gA |-> Frac (fA /. (mrows * mcols * (bm/tm * bn))) eA **
  gB |-> Frac (fB /. (mrows * mcols * (bm/tm * bn))) eB **
  forall+ (i : natlt tm).
    exists* v. // NO FUNCTIONAL SPEC RIGHT NOW
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
      ((tid / bn * tm) + i)
      (tid % bn)
      v

(* The barrier flip-flops between an initial state
where every threads shares all of the two arrays, and
a second state where every thread owns a single cell
in each array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)
let own_1_cell
  (#et : Type0)
  (#rows #cols : szp)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (i : natlt rows)
  (j : natlt cols)
  : slprop =
  exists* va. gpu_matrix_pts_to_cell m i j va

let barrier_p
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  : B.barrier_side (bm/tm * bn) =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x)
    else
      own_1_cell m1 (tid/bk) (tid%bk) ** own_1_cell m2 (tid/bn) (tid%bn)

let barrier_q
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  : B.barrier_side (bm/tm * bn) =
  fun it tid -> barrier_p tm m1 m2 (it+1) tid (* flip flop *)

let barrier_contract
  (#et : Type0)
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2) : B.contract (bm/tm * bn) = {
    rin = barrier_p tm m1 m2;
    rout = barrier_q tm m1 m2;
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
  kpost1 comb tm gA gB gC eA eB eC fA fB bid tid **
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
    gB |-> Frac fB eB **
    own_1_cell sA (tid/bk) (tid%bk) **
    own_1_cell sB (tid/bn) (tid%bn)
{
  unfold own_1_cell sA (tid/bk) (tid%bk);
  let tA = gpu_matrix_extract_tile_ro' gA (SZ.v bm) (SZ.v bk) (SZ.v mrow) (SZ.v mk);
  let va = M.gpu_matrix_read tA (tid /^ bk) (tid %^ bk);
  M.gpu_matrix_write_cell sA (tid /^ bk) (tid %^ bk) va;
  Trade.elim_trade _ _;
  fold own_1_cell sA (tid/bk) (tid%bk);

  unfold own_1_cell sB (tid/bn) (tid%bn);
  let tB = gpu_matrix_extract_tile_ro' gB (SZ.v bk) (SZ.v bn) (SZ.v mk) (SZ.v mcol);
  let vb = M.gpu_matrix_read tB (tid /^ bn) (tid %^ bn);
  M.gpu_matrix_write_cell sB (tid /^ bn) (tid %^ bn) vb;
  Trade.elim_trade _ _;
  fold own_1_cell sB (tid/bn) (tid%bn);
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
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
    B.barrier_tok (barrier_contract tm (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid
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
         as barrier_p tm sA sB (2 * !bkIdx) tid;
    rewrite barrier_p tm sA sB (2 * !bkIdx) tid
         as (barrier_contract tm sA sB).rin (2 * !bkIdx) tid;

    B.barrier_wait ();

    rewrite (barrier_contract tm sA sB).rout (2 * !bkIdx) tid
         as barrier_q tm sA sB (2 * !bkIdx) tid;
    rewrite barrier_q tm sA sB (2 * !bkIdx) tid
         as own_1_cell sA (tid/bk) (tid%bk) ** own_1_cell sB (tid/bn) (tid%bn);

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    populate_shmem tm sA sB gA gB mrow !bkIdx mcol tid;

    odd_2x1 !bkIdx;
    assert pure (odd (2 * !bkIdx + 1));
    rewrite own_1_cell sA (tid/bk) (tid%bk) ** own_1_cell sB (tid/bn) (tid%bn)
         as barrier_p tm sA sB (2 * !bkIdx + 1) tid;
    rewrite barrier_p tm sA sB (2 * !bkIdx + 1) tid
         as (barrier_contract tm sA sB).rin (2 * !bkIdx + 1) tid;

    B.barrier_wait ();
    even_2x (!bkIdx + 1);
    (* sigh *)
    let vbkIdx = !bkIdx;
    assert (pure (2 * (vbkIdx + 1) == 2 * vbkIdx + 1 + 1));
    assert (pure (even (2 * vbkIdx + 2)));
    rewrite (barrier_contract tm sA sB).rout (2 * vbkIdx + 1) tid
         as barrier_q tm sA sB (2 * vbkIdx + 1) tid;
    rewrite barrier_q tm sA sB (2 * vbkIdx + 1) tid
    as
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * bn)) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * bn)) x);

    subproducts1d tm cache1d sA sB threadRow threadCol;

    (* Move to next tile *)
    bkIdx := !bkIdx +^ 1sz;
  };

  // Weaken the ownership of C
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

  M.gpu_matrix_concr sA; rewrite each M.core sA as sarA;
  M.gpu_matrix_concr sB; rewrite each M.core sB as sarB;

  rewrite each sA as M.from_array slA sarA;
  rewrite each sB as M.from_array slB sarB;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  drop_ (B.barrier_tok _);
  drop_ (B.barrier_state _);

  ();
}
#pop-options


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
  admit();
}

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
  admit();
}

ghost
fn block_teardown
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
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost1 comb tm gA gB gC eA eB eC fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
      kpost1 comb tm gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  // forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
  //   (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}

#push-options "--fuel 2 --ifuel 2 --z3rlimit_factor 10 --z3refresh"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk = mrows *^ mcols; //SZ.uint_to_t (SZ.v mrows * SZ.v mcols);
  nthr = (bm /^ tm *^ bn);

  shmems_desc = shmems_desc et bm bn bk;

  barrier_contract = (fun bid ptrs -> barrier_contract tm (M.from_array slA (fst ptrs)) (M.from_array slB (fst (snd ptrs))));
  barrier_ok = (fun bid ptrs -> magic());

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpre1  comb tm gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpost1 comb tm gA gB gC eA eB eC fA fB bid tid);
  setup      = setup    comb tm gA gB gC #eA #eB #eC;
  teardown   = teardown comb tm gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    =  block_setup    comb tm slA slB gA gB gC #eA #eB #eC;
  block_teardown = block_teardown comb tm slA slB gA gB gC #eA #eB #eC;

  kpre      = kpre  comb tm slA slB gA gB gC eA eB eC fA fB;
  kpost     = kpost comb tm slA slB gA gB gC eA eB eC fA fB;

  f = kf comb tm slA slB gA gB gC #fA #fB #eA #eB;

  block_pre_sendable=magic();
  block_post_sendable=magic();
  kpre_sendable=magic();
  kpost_sendable=magic()
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb tm (R.row_major _ _) (R.row_major _ _) gA #fA gB #fB gC #eA #eB #eC ());
}
