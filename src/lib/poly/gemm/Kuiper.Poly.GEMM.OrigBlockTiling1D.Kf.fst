module Kuiper.Poly.GEMM.OrigBlockTiling1D.Kf

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module B = Kuiper.Barrier
module M = Kuiper.Matrix
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade

open Kuiper.Poly.GEMM.Copy { live_cell }
open Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs

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
    decreases (bk - !dotIdx)
  {
    let tmpB = M.gpu_matrix_read gB !dotIdx bcol;
    let mut resIdx = 0sz;
    while (!resIdx <^ tm)
      invariant live resIdx ** pure (!resIdx <= tm)
      invariant live rch1d
      decreases (tm - !resIdx)
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
#restart-solver
#push-options "--z3rlimit 200 --ifuel 1"
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
    decreases (mshared - !bkIdx)
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
    decreases (tm - !resIdx)
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
