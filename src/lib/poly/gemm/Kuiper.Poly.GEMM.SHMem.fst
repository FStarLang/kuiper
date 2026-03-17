module Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.View.TwoTiles { aview_2tile2, mkAIdx, mkCIdx }

module M = Kuiper.Matrix
open Kuiper.Matrix {
  gpu_matrix,
  gpu_matrix_pts_to,
  gpu_matrix_pts_to_cell,
  is_global_matrix
}

module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix { ematrix, macc }
open Kuiper.Poly.GEMM.Copy { live_cell }

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |} (tile:valid_tile) : list shmem_desc = [
  SHArray et (tile *^ tile);
  SHArray et (tile *^ tile);
]

(* Per-cell barrier contract: threads flip-flop between shared fractional
   ownership (even steps) and exclusive single-cell ownership (odd steps).
   This directly matches the kernel's access pattern where thread tid writes
   cell (tid/tile, tid%tile) in each shmem matrix.

   Even steps (rin): each thread gives back fractional read ownership
   Odd steps (rin): each thread gives back single-cell write ownership
   Even steps (rout): each thread receives single-cell write ownership
   Odd steps (rout): each thread receives fractional read ownership
     with *specific* content (the correct subtile) so the approximation
     proof goes through. *)
let barrier_p_cell
  (#et : Type0)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_mlayout tile tile)
  (sa1 : gpu_matrix et slA)
  (sa2 : gpu_matrix et slB)
  : B.barrier_side (tile * tile)
  = fun it tid ->
    if it >= 2 * mshared then
      emp
    else if even it then
      (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
      (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
    else
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gpu_matrix_pts_to_cell sa1 (tid/tile) (tid%tile) (macc (ematrix_subtile eA tile tile mrow (it / 2)) (tid/tile) (tid%tile)) **
      gpu_matrix_pts_to_cell sa2 (tid/tile) (tid%tile) (macc (ematrix_subtile eB tile tile (it / 2) mcol) (tid/tile) (tid%tile))

let barrier_q_cell
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_mlayout tile tile)
  (sa1 : gpu_matrix et slA)
  (sa2 : gpu_matrix et slB)
  : B.barrier_side (tile * tile)
  = fun it tid ->
    if it >= 2 * mshared then
      emp
    else if even it then
      live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile)
    else
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      sa1 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eA tile tile mrow (it / 2)) **
      sa2 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eB tile tile (it / 2) mcol)

let shmem_contract
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_mlayout tile tile)
  (sa1 : gpu_matrix et slA)
  (sa2 : gpu_matrix et slB)
  : B.contract (tile * tile) = {
    rin  = barrier_p_cell tile eA eB bid sa1 sa2;
    rout = barrier_q_cell tile eA eB bid sa1 sa2;
  }

#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
ghost
fn barrier_p_to_q_cell_transform
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_mlayout tile tile)
  (sa1 : gpu_matrix et slA)
  (sa2 : gpu_matrix et slB)
  (#_ : squash (SZ.fits (mlayout_size slA)))
  (#_ : squash (SZ.fits (mlayout_size slB)))
  (it : nat)
  requires
    forall+ (tid : natlt (tile * tile)).
      barrier_p_cell tile eA eB bid sa1 sa2 it tid
  ensures
    forall+ (tid : natlt (tile * tile)).
      barrier_q_cell tile eA eB bid sa1 sa2 it tid
{
  if (it >= 2 * mshared) {
    forevery_map
      (fun (tid : natlt (tile * tile)) -> barrier_p_cell tile eA eB bid sa1 sa2 it tid)
      (fun (tid : natlt (tile * tile)) -> barrier_q_cell tile eA eB bid sa1 sa2 it tid)
      fn tid {
        rewrite barrier_p_cell tile eA eB bid sa1 sa2 it tid as emp;
        rewrite emp as barrier_q_cell tile eA eB bid sa1 sa2 it tid;
      };
  } else {
    let ev = even it;
    if ev {
      assert pure (it < 2 * mshared);
      assert pure (even it);
      forevery_map
        (fun (tid : natlt (tile * tile)) -> barrier_p_cell tile eA eB bid sa1 sa2 it tid)
        (fun (tid : natlt (tile * tile)) ->
          (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
          (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x))
        fn tid {
          rewrite barrier_p_cell tile eA eB bid sa1 sa2 it tid
               as (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
                  (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x);
        };
      forevery_unzip _ _;
      M.gpu_matrix_gather_n_underspec sa1 (tile * tile);
      with em1. assert sa1 |-> em1;
      M.gpu_matrix_explode sa1;
      forevery_unfactor' (tile * tile) tile tile
        (fun r c -> gpu_matrix_pts_to_cell sa1 r c (macc em1 r c));
      forevery_map
        (fun (tid : natlt (tile * tile)) -> gpu_matrix_pts_to_cell sa1 (tid/tile) (tid%tile) (macc em1 (tid/tile) (tid%tile)))
        (fun (tid : natlt (tile * tile)) -> live_cell sa1 (tid/tile) (tid%tile))
        fn tid { fold (live_cell sa1 (tid/tile) (tid%tile)) };
      M.gpu_matrix_gather_n_underspec sa2 (tile * tile);
      with em2. assert sa2 |-> em2;
      M.gpu_matrix_explode sa2;
      forevery_unfactor' (tile * tile) tile tile
        (fun r c -> gpu_matrix_pts_to_cell sa2 r c (macc em2 r c));
      forevery_map
        (fun (tid : natlt (tile * tile)) -> gpu_matrix_pts_to_cell sa2 (tid/tile) (tid%tile) (macc em2 (tid/tile) (tid%tile)))
        (fun (tid : natlt (tile * tile)) -> live_cell sa2 (tid/tile) (tid%tile))
        fn tid { fold (live_cell sa2 (tid/tile) (tid%tile)) };
      forevery_zip
        (fun (tid : natlt (tile * tile)) -> live_cell sa1 (tid/tile) (tid%tile)) _;
      forevery_map
        (fun (tid : natlt (tile * tile)) -> live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile))
        (fun (tid : natlt (tile * tile)) -> barrier_q_cell tile eA eB bid sa1 sa2 it tid)
        fn tid {
          rewrite live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile)
               as barrier_q_cell tile eA eB bid sa1 sa2 it tid;
        };
    } else {
      assert pure (it < 2 * mshared);
      assert pure (odd it);
      let mrow = bid / mcols;
      let mcol = bid % mcols;
      forevery_map
        (fun (tid : natlt (tile * tile)) -> barrier_p_cell tile eA eB bid sa1 sa2 it tid)
        (fun (tid : natlt (tile * tile)) ->
          gpu_matrix_pts_to_cell sa1 (tid/tile) (tid%tile) (macc (ematrix_subtile eA tile tile mrow (it/2)) (tid/tile) (tid%tile)) **
          gpu_matrix_pts_to_cell sa2 (tid/tile) (tid%tile) (macc (ematrix_subtile eB tile tile (it/2) mcol) (tid/tile) (tid%tile)))
        fn tid {
          rewrite barrier_p_cell tile eA eB bid sa1 sa2 it tid
               as gpu_matrix_pts_to_cell sa1 (tid/tile) (tid%tile) (macc (ematrix_subtile eA tile tile mrow (it/2)) (tid/tile) (tid%tile)) **
                  gpu_matrix_pts_to_cell sa2 (tid/tile) (tid%tile) (macc (ematrix_subtile eB tile tile (it/2) mcol) (tid/tile) (tid%tile));
        };
      forevery_unzip _ _;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> gpu_matrix_pts_to_cell sa1 r c (macc (ematrix_subtile eA tile tile mrow (it/2)) r c));
      M.gpu_matrix_implode sa1;
      M.gpu_matrix_share_n sa1 (tile * tile);
      forevery_factor' (tile * tile) tile tile
        (fun r c -> gpu_matrix_pts_to_cell sa2 r c (macc (ematrix_subtile eB tile tile (it/2) mcol) r c));
      M.gpu_matrix_implode sa2;
      M.gpu_matrix_share_n sa2 (tile * tile);
      forevery_zip
        (fun (_ : natlt (tile * tile)) -> sa1 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eA tile tile mrow (it/2))) _;
      forevery_map
        (fun (tid : natlt (tile * tile)) ->
          sa1 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eA tile tile mrow (it/2)) **
          sa2 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eB tile tile (it/2) mcol))
        (fun (tid : natlt (tile * tile)) -> barrier_q_cell tile eA eB bid sa1 sa2 it tid)
        fn tid {
          rewrite
            sa1 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eA tile tile mrow (it/2)) **
            sa2 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eB tile tile (it/2) mcol)
          as
            barrier_q_cell tile eA eB bid sa1 sa2 it tid;
        };
    }
  }
}
#pop-options

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  (gA |-> Frac (fA /. mlayout_vsize lC) eA) **
  (gB |-> Frac (fB /. mlayout_vsize lC) eB) **
  gpu_matrix_pts_to_cell
    (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
    #1.0R
    (tid / tile) (tid % tile) (macc eC grow gcol)

(* Functional postcondition: the cell contains a value approximating
   MS.gemm_single over external real matrices rA, rB, rC *)
unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  (gA |-> Frac (fA /. mlayout_vsize lC) eA) **
  (gB |-> Frac (fB /. mlayout_vsize lC) eB) **
  (exists* (v : et).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      #1.0R
      (tid / tile) (tid % tile) v **
    pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol))

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x)

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x)

(* TODO: Find out where the time is going when checking this function,
it feels a lot slower than the others. *)
#push-options "--z3rlimit 200"
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows * tile)   (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile)   (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (#_sq : squash (eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : szlt (mrows * mcols))
  (tid : szlt (tile  * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (shmem_contract tile eA eB bid (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (shmem_contract tile eA eB bid (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state (2 * mshared)
{
  let (ar1, (ar2, _)) = sh;

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  M.gpu_matrix_abs' slA ar1;
  let sa1 = M.from_array slA ar1;
  rewrite each M.from_array slA ar1 as sa1;

  M.gpu_matrix_abs' slB ar2;
  let sa2 = M.from_array slB ar2;
  rewrite each M.from_array slB ar2 as sa2;

  let gTile = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  rewrite each gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols) as gTile;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (SZ.v brow == tid / tile));
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  rewrite each (tid / tile) as v brow;
  rewrite each (tid % tile) as v bcol;

  let mut sum : et = zero;
  let mut bk  : sz = 0sz;

  let grow : erased (natlt (mrows * tile)) = hide (SZ.v mrow * SZ.v tile + SZ.v brow);
  let gcol : erased (natlt (mcols * tile)) = hide (SZ.v mcol * SZ.v tile + SZ.v bcol);

  (* Introduce the pure approximation facts from the squash parameter *)
  assert pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC);

  while (SZ.(!bk <^ mshared))
    invariant
      exists* (vbk : SZ.t{vbk <= mshared}) sumv.
        bk |-> vbk **
        sum |-> sumv **
        B.barrier_state (2 * vbk) **
        pure (v_approximates sumv (MS.__gmatmul_single 0.0R ( *. ) ( +. ) rA rB grow gcol (SZ.v vbk * SZ.v tile)))
    invariant
      (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
      (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
  {
    let vbk = !bk;
    gpu_matrix_extract_tile_ro gA tile tile mrow vbk;
    let aTile = gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v vbk);
    assert (rewrites_to aTile (gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v vbk)));
    gpu_matrix_extract_tile_ro gB tile tile vbk mcol;
    let bTile = gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v mcol);
    assert (rewrites_to bTile (gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v mcol)));

    let v1 = M.gpu_matrix_read aTile brow bcol;
    let v2 = M.gpu_matrix_read bTile brow bcol;

    ambig_trade_elim ();
    ambig_trade_elim ();

    even_2x !bk;

    (* Even barrier: give shared ownership, receive per-cell ownership *)
    rewrite (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
            (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
         as barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbk) tid;
    rewrite barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbk) tid
         as (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rin (2 * vbk) tid;

    B.barrier_wait ();

    rewrite (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rout (2 * vbk) tid
         as barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbk) tid;
    rewrite barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbk) tid
         as live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile);
    rewrite each (tid / tile) as v brow;
    rewrite each (tid % tile) as v bcol;

    (* Write to shmem: unfold live_cell, write, keep specific content *)
    unfold live_cell sa1 (v brow) (v bcol);
    M.gpu_matrix_write_cell sa1 brow bcol v1;

    unfold live_cell sa2 (v brow) (v bcol);
    M.gpu_matrix_write_cell sa2 brow bcol v2;

    (* Odd barrier: give per-cell ownership with specific content *)
    odd_2x1 !bk;
    assert (pure (odd (2 * !bk + 1)));

    rewrite each (v brow) as (tid / tile);
    rewrite each (v bcol) as (tid % tile);
    rewrite each v1 as (macc (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v vbk)) (tid / tile) (tid % tile));
    rewrite each v2 as (macc (ematrix_subtile eB tile tile (SZ.v vbk) (SZ.v mcol)) (tid / tile) (tid % tile));
    rewrite
      gpu_matrix_pts_to_cell sa1 (tid / tile) (tid % tile)
        (macc (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v vbk)) (tid / tile) (tid % tile)) **
      gpu_matrix_pts_to_cell sa2 (tid / tile) (tid % tile)
        (macc (ematrix_subtile eB tile tile (SZ.v vbk) (SZ.v mcol)) (tid / tile) (tid % tile))
    as barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbk + 1) tid;
    rewrite barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbk + 1) tid
         as (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rin (2 * vbk + 1) tid;

    B.barrier_wait ();

    even_2x (!bk + 1);
    assert pure (2 * (!bk + 1) == 2 * !bk + 2);
    assert pure (odd (2 * !bk + 1));
    assert pure (even (2 * !bk + 2));
    let vbkIdx = !bk;
    assert pure (SZ.v vbkIdx < mshared);
    assert pure ((2 * SZ.v vbkIdx + 1) < 2 * mshared);
    assert pure ((2 * SZ.v vbkIdx + 1) / 2 == SZ.v vbkIdx);

    (* After the odd barrier, barrier_q_cell returns specific content:
       the shmem matrices point to the correct subtiles of eA/eB.
       This is critical for the approximation proof below. *)
    rewrite (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rout (2 * vbkIdx + 1) tid
         as barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbkIdx + 1) tid;
    rewrite barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * vbkIdx + 1) tid
         as sa1 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v vbkIdx)) **
            sa2 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eB tile tile (SZ.v vbkIdx) (SZ.v mcol));

    rewrite each (tid / tile) as v brow;
    rewrite each (tid % tile) as v bcol;

    (* At this point the SHMem cache is filled with the correct subtiles
       and we have RO permission with specific content. Compute product for
       our cell in the tile and add to sum. *)

    (* matmul_dotprod now gets eA/eB subtile content from the barrier,
       so t == MS.matmul_single (subtile eA) (subtile eB) brow bcol. *)
    let t = Kuiper.Poly.GEMM.Util.matmul_dotprod sa1 sa2 brow bcol;
    let s = !sum;
    sum := s `add` t;

    (* Prove the approximation invariant is maintained:
       t == matmul_single (subtile eA) (subtile eB) brow bcol,
       so t %~ matmul_single (subtile rA) (subtile rB) brow bcol
       (by __matmul_single_approx_real).
       Combined with s %~ __gmatmul_single ... rA rB grow gcol (vbk * tile)
       and the step lemma (__gmatmul_single_split), we get
       s+t %~ __gmatmul_single ... rA rB grow gcol ((vbk+1) * tile). *)
    let sub_rA = ematrix_subtile rA tile tile (SZ.v mrow) (SZ.v vbkIdx);
    let sub_rB = ematrix_subtile rB tile tile (SZ.v vbkIdx) (SZ.v mcol);
    MU.__matmul_single_approx_real
      (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v vbkIdx))
      (ematrix_subtile eB tile tile (SZ.v vbkIdx) (SZ.v mcol))
      sub_rA sub_rB
      brow bcol tile;

    let r_partial = MS.__gmatmul_single 0.0R ( *. ) ( +. ) rA rB grow gcol (SZ.v vbkIdx * SZ.v tile);
    let r_subtile = MS.__gmatmul_single 0.0R ( *. ) ( +. ) sub_rA sub_rB brow bcol tile;

    (* Step the partial sum: split property of __gmatmul_single over rA, rB *)
    MU.__gmatmul_single_split rA rB grow gcol (SZ.v vbkIdx * SZ.v tile) tile sub_rA sub_rB brow bcol;
    assert (pure (
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) rA rB grow gcol (SZ.v vbkIdx * SZ.v tile + SZ.v tile)
      == r_partial +. r_subtile));
    assert (pure ((SZ.v vbkIdx + 1) * SZ.v tile == SZ.v vbkIdx * SZ.v tile + SZ.v tile));

    a_add s t r_partial r_subtile;

    // What the hell.
    assert (pure (2 * (!bk + 1) == 2 * !bk + 1 + 1));

    (* Move to next tile *)
    bk := !bk +^ 1sz;
    ()
  };

  let s = !sum;
  (* After the loop, vbk == mshared, so:
     s %~ __gmatmul_single ... rA rB grow gcol (mshared * tile)
        == MS.matmul_single rA rB grow gcol *)

  let v0 = M.gpu_matrix_read_cell gTile brow bcol;
  let v1 = comb v0 s;
  M.gpu_matrix_write_cell gTile brow bcol v1;

  (* v0 == macc eC grow gcol, and eC %~ rC, so v0 %~ macc rC grow gcol.
     s %~ MS.matmul_single rA rB grow gcol.
     v1 = comb v0 s.
     By approx2 comb comb_r: v1 %~ comb_r (macc rC grow gcol) (MS.matmul_single rA rB grow gcol)
     Which is: v1 %~ MS.gemm_single comb_r rA rB rC grow gcol *)

  with v'.
    rewrite
      M.gpu_matrix_pts_to_cell gTile brow bcol v'
    as
      M.gpu_matrix_pts_to_cell gTile
        (tid / tile) (tid % tile) v';

  M.gpu_matrix_concr sa1; rewrite each M.core sa1 as ar1;
  M.gpu_matrix_concr sa2; rewrite each M.core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  rewrite each gTile as gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);

  fold (kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  ()
}
#pop-options

#push-options "--z3rlimit 100"
ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA #eB : ematrix _ _ _)
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  (* Step 1: Share gA/gB, explode gC *)
  M.gpu_matrix_share_n gA (mlayout_vsize lC);
  M.gpu_matrix_share_n gB (mlayout_vsize lC);
  gpu_matrix_explode_tiled gC (SZ.v tile) (SZ.v tile);
  forevery_rw_size4 ((mrows * tile) / tile) mrows ((mcols * tile) / tile) mcols (SZ.v tile) tile (SZ.v tile) tile;

  (* Step 2: Factor gA/gB to 2D *)
  forevery_factor (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile) (fun _ -> gA |-> Frac (fA /. mlayout_vsize lC) eA);
  forevery_factor (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile) (fun _ -> gB |-> Frac (fB /. mlayout_vsize lC) eB);

  (* Step 3: Convert 4D -> 2D for gC *)
  assert pure (forall (mrow:natlt mrows) (mcol:natlt mcols). (mrow * mcols + mcol) / mcols == mrow /\ (mrow * mcols + mcol) % mcols == mcol);
  assert pure (forall (brow:natlt tile) (bcol:natlt tile). (brow * tile + bcol) / tile == brow /\ (brow * tile + bcol) % tile == bcol);
  forevery_ext_4
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol) brow bcol (macc eC (mrow * tile + brow) (mcol * tile + bcol)))
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      let bid = mrow * mcols + mcol in let tid = brow * tile + bcol in
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  forevery_unfactor_2 (mrows * mcols) mrows mcols (SZ.v tile * SZ.v tile) (SZ.v tile) (SZ.v tile)
    (fun bid tid -> gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
      (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 4: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gA |-> Frac (fA /. mlayout_vsize lC) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gB |-> Frac (fB /. mlayout_vsize lC) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 5: Bridge to natlt2 and match kpre1 *)
  forevery_rw_size2 (mrows * mcols) (SZ.v (mrows `SZ.mul` mcols)) (SZ.v tile * SZ.v tile) (SZ.v (tile `SZ.mul` tile));
  forevery_ext_2
    (fun (bid : natlt (SZ.v (mrows `SZ.mul` mcols))) (tid : natlt (SZ.v (tile `SZ.mul` tile))) ->
      gA |-> Frac (fA /. mlayout_vsize lC) eA ** gB |-> Frac (fB /. mlayout_vsize lC) eB **
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))))
    (fun (bid : natlt2 mrows mcols) (tid : natlt2 tile tile) -> kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ();
}
#pop-options

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt2 tile  tile).
      kpre comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  // Share shmem into (tile * tile) copies
  gpu_live_c_shmems_share_underspec sh #1.0R #(tile * tile);

  // Unfold live_c_shmems into explicit gpu_pts_to_array inside forall+
  forevery_map
    (fun (_ : natlt (tile * tile)) -> live_c_shmems sh #(1.0R /. (tile * tile)))
    (fun (_ : natlt (tile * tile)) ->
      (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
      (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x))
    fn _ {
      unfold_live_c_shmems_cons sh #(1.0R /. (tile * tile));
      unfold_live_c_shmem (fst sh) #(1.0R /. (tile * tile));
      unfold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
      unfold_live_c_shmem (fst (snd sh)) #(1.0R /. (tile * tile));
      unfold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));
    };

  // Bridge natlt type: natlt (tile * tile) → natlt2 tile tile
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));

  // Zip shmem fracs with kpre1 to form kpre
  forevery_zip
    (fun (tid : natlt2 tile tile) ->
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;
}

#push-options "--z3rlimit 20"
ghost
fn block_teardown
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    (forall+ (tid : natlt2 tile  tile).
      kpost comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
{
  // Convert natlt2 → natlt (tile * tile) first (1 forall+, no ambiguity)
  forevery_rw_size (SZ.v (tile *^ tile)) (tile * tile);

  // Unzip kpost into kpost1 + shmem fracs
  forevery_unzip
    (fun (tid : natlt (tile * tile)) ->
      kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
    _;

  // Fold explicit gpu_pts_to_array back into live_c_shmems inside forall+
  forevery_map
    (fun (_ : natlt (tile * tile)) ->
      (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
      (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x))
    (fun (_ : natlt (tile * tile)) -> live_c_shmems sh #(1.0R /. (tile * tile)))
    fn _ {
      fold_live_c_shmem (fst (snd sh)) #(1.0R /. (tile * tile));
      fold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));
      fold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
      fold_live_c_shmem (fst sh) #(1.0R /. (tile * tile));
      fold_live_c_shmems_cons sh #(1.0R /. (tile * tile));
    };

  // Gather shmem fractions back
  gpu_live_c_shmems_gather_underspec sh #1.0R #(tile * tile);

  // Convert kpost1 back to natlt2 (1 forall+, no ambiguity)
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));
}
#pop-options

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
      gC |-> eC' **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  (* Step 1: Bridge natlt2 → natlt *)
  forevery_rw_size2
    (SZ.v (mrows *^ mcols)) (mrows * mcols)
    (SZ.v (tile *^ tile))   (SZ.v tile * SZ.v tile);

  (* Step 2: Unfold kpost1 explicitly *)
  forevery_ext_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gA |-> Frac (fA /. mlayout_vsize lC) eA **
      gB |-> Frac (fB /. mlayout_vsize lC) eB **
      (exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. mlayout_vsize lC) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gB |-> Frac (fB /. mlayout_vsize lC) eB **
      (exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. mlayout_vsize lC) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. mlayout_vsize lC) eA);
  M.gpu_matrix_gather_n gA (mlayout_vsize lC);
  forevery_unfactor' (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. mlayout_vsize lC) eB);
  M.gpu_matrix_gather_n gB (mlayout_vsize lC);

  (* Step 6: Collect gC cells back into matrix *)
  let _vf = gpu_matrix_collect_approx_tiled gC (SZ.v tile) (SZ.v tile)
    (SZ.v mrows) (SZ.v mcols)
    (fun (row : natlt (mrows * tile)) (col : natlt (mcols * tile)) (v : et) ->
      v %~ MS.gemm_single comb_r rA rB rC row col);

  (* Step 7: Prove eC' %~ MS.mmcomb comb_r rC rA rB *)
  with eC'. assert (gC |-> eC');
  assert pure (eC' %~ MS.mmcomb comb_r rC rA rB);
  ();
}

#push-options "--z3rlimit_factor 10 --fuel 0 --ifuel 0 --split_queries no"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (#_ : squash (SZ.fits (mlayout_size slA)))
  (#_ : squash (SZ.fits (mlayout_size slB)))
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads
               /\ eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
          gC |-> eC' **
          pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  barrier_contract = (fun _bid ptrs -> shmem_contract tile eA eB _bid (M.from_array slA (fst ptrs)) (M.from_array slB (fst (snd ptrs))));
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun _bid _ptrs -> barrier_p_to_q_cell_transform tile eA eB _bid (M.from_array slA (fst _ptrs)) (M.from_array slB (fst (snd _ptrs))));

  shmems_desc = shmems_desc et tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre1  comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup      = setup    tile comb comb_r gA gB gC;
  teardown   = teardown tile comb comb_r gA gB gC rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile slA slB comb comb_r gA gB gC #_ #_ #_ #_ #eC;
  block_teardown = block_teardown tile slA slB comb comb_r gA gB gC #_ #_ #_ #_ #eC rA rB rC;

  kpre      = kpre  comb comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;

  f = kf tile slA slB comb comb_r gA gB gC rA rB rC;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=magic();
  kpost_sendable=magic();
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (lA : mlayout (mrows   * tile) (mshared * tile))
  (lB : mlayout (mshared * tile) (mcols   * tile))
  (lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (rA  : ematrix real (mrows * tile) (mshared * tile))
  (rB  : ematrix real (mshared * tile) (mcols * tile))
  (rC  : ematrix real (mrows * tile) (mcols * tile))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile (R.row_major _ _) (R.row_major _ _) comb comb_r gA gB gC rA rB rC ());
}
#pop-options
