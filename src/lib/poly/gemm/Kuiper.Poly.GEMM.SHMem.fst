module Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor.Tiling
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Array2 { array2 }
module M = Kuiper.Array2
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

open Kuiper.EMatrix { ematrix, macc }

let live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : M.layout rows cols)
  (gm : array2 et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. M.pts_to_cell gm (i, j) v

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |} (tile:valid_tile) : list shmem_desc = [
  SHArray et (tile *^ tile);
  SHArray et (tile *^ tile);
]

(* Helper: bridge between pair-indexed forall+ (from M.explode)
   and 2-arg indexed forall+ (for forevery_unfactor' etc.) *)
ghost
fn explode2
  (#et : Type0) (#rows #cols : nat) (#l : M.layout rows cols)
  (a : array2 et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires a |-> Frac f s
  ensures
    forall+ (r : natlt rows) (c : natlt cols).
      M.pts_to_cell a #f (r, c) (macc s r c)
{
  M.explode a;
  forevery_ext _ (fun i -> M.pts_to_cell a #f (fst i, snd i) (macc s (fst i) (snd i)));
  forevery_unflatten (fun r c -> M.pts_to_cell a #f (r, c) (macc s r c));
  ()
}

ghost
fn implode2
  (#et : Type0) (#rows #cols : nat) (#l : M.layout rows cols)
  (a : array2 et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (r : natlt rows) (c : natlt cols).
      M.pts_to_cell a #f (r, c) (macc s r c)
  ensures a |-> Frac f s
{
  forevery_flatten (fun r c -> M.pts_to_cell a #f (r, c) (macc s r c));
  forevery_ext _ (fun i -> M.pts_to_cell a #f i (macc s (fst i) (snd i)));
  M.implode a;
  ()
}

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
  (#slA #slB : M.full_layout tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
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
      M.pts_to_cell sa1 (tid/tile, tid%tile) (macc (ematrix_subtile eA tile tile mrow (it / 2)) (tid/tile) (tid%tile)) **
      M.pts_to_cell sa2 (tid/tile, tid%tile) (macc (ematrix_subtile eB tile tile (it / 2) mcol) (tid/tile) (tid%tile))

let barrier_q_cell
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : M.full_layout tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
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
  (#slA #slB : M.full_layout tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
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
  (#slA #slB : M.full_layout tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
  (#_ : squash (SZ.fits (M.layout_size slA)))
  (#_ : squash (SZ.fits (M.layout_size slB)))
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
      M.gather_n_underspec sa1 (tile * tile);
      M.gather_n_underspec sa2 (tile * tile);
      with em1. assert (pts_to sa1 #1.0R em1);
      with em2. assert (pts_to sa2 #1.0R em2);
      explode2 sa1;
      forevery_unfactor' (tile * tile) tile tile
        (fun r c -> M.pts_to_cell sa1 (r, c) (macc em1 r c));
      explode2 sa2;
      forevery_unfactor' (tile * tile) tile tile
        (fun r c -> M.pts_to_cell sa2 (r, c) (macc em2 r c));
      forevery_map
        (fun (tid : natlt (tile * tile)) -> M.pts_to_cell sa1 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc em1 (tid/tile) (tid%tile)))
        (fun (tid : natlt (tile * tile)) -> live_cell sa1 (tid/tile) (tid%tile))
        fn tid { fold live_cell };
      forevery_map
        (fun (tid : natlt (tile * tile)) -> M.pts_to_cell sa2 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc em2 (tid/tile) (tid%tile)))
        (fun (tid : natlt (tile * tile)) -> live_cell sa2 (tid/tile) (tid%tile))
        fn tid { fold live_cell };
      forevery_zip
        (fun (tid : natlt (tile * tile)) -> live_cell sa1 (tid/tile) (tid%tile))
        (fun (tid : natlt (tile * tile)) -> live_cell sa2 (tid/tile) (tid%tile));
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
          M.pts_to_cell sa1 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc (ematrix_subtile eA tile tile mrow (it/2)) (tid/tile) (tid%tile)) **
          M.pts_to_cell sa2 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc (ematrix_subtile eB tile tile (it/2) mcol) (tid/tile) (tid%tile)))
        fn tid {
          rewrite barrier_p_cell tile eA eB bid sa1 sa2 it tid
               as M.pts_to_cell sa1 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc (ematrix_subtile eA tile tile mrow (it/2)) (tid/tile) (tid%tile)) **
                  M.pts_to_cell sa2 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc (ematrix_subtile eB tile tile (it/2) mcol) (tid/tile) (tid%tile));
        };
      forevery_unzip _ _;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> M.pts_to_cell sa1 (Mktuple2 #(natlt tile) #(natlt tile) r c) (macc (ematrix_subtile eA tile tile mrow (it/2)) r c));
      implode2 sa1;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> M.pts_to_cell sa2 (Mktuple2 #(natlt tile) #(natlt tile) r c) (macc (ematrix_subtile eB tile tile (it/2) mcol) r c));
      implode2 sa2;
      M.share_n sa1 (tile * tile);
      M.share_n sa2 (tile * tile);
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
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  (gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA) **
  (gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB) **
  M.pts_to_cell
    (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
    #1.0R
    (tid / tile, tid % tile) (macc eC grow gcol)

(* Functional postcondition: the cell contains a value approximating
   MS.gemm_single over external real matrices rA, rB, rC *)
unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  (gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA) **
  (gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB) **
  (exists* (v : et).
    M.pts_to_cell
      (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      #1.0R
      (tid / tile, tid % tile) v **
    pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol))

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
  live_c_shmems sh #(1.0R /. (tile * tile))

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  live_c_shmems sh #(1.0R /. (tile * tile))

(* TODO: Find out where the time is going when checking this function,
it feels a lot slower than the others. *)
#push-options "--z3rlimit 200 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  (* Unfold live_c_shmems to get raw gpu_pts_to_array *)
  unfold_live_c_shmems_cons sh #(1.0R /. (tile * tile));
  unfold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
  unfold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));

  let (ar1, (ar2, _)) = sh;

  rewrite (live_c_shmem ar1 #(1.0R /. (tile * tile)))
      as  (exists* v. gpu_pts_to_array ar1 #(1.0R /. (tile * tile)) v);
  rewrite (live_c_shmem ar2 #(1.0R /. (tile * tile)))
      as  (exists* v. gpu_pts_to_array ar2 #(1.0R /. (tile * tile)) v);

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  M.raise' slA ar1;
  let sa1 = M.from_array slA ar1;
  rewrite each M.from_array slA ar1 as sa1;

  M.raise' slB ar2;
  let sa2 = M.from_array slB ar2;
  rewrite each M.from_array slB ar2 as sa2;

  let gTile = array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  rewrite each array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols) as gTile;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (SZ.v brow == tid / tile));
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  rewrite each (tid / tile) as v brow;
  rewrite each (tid % tile) as v bcol;

  let mut sum : et = zero;
  let mut bk  : szle mshared = 0sz;

  let grow : erased (natlt (mrows * tile)) = hide (SZ.v mrow * SZ.v tile + SZ.v brow);
  let gcol : erased (natlt (mcols * tile)) = hide (SZ.v mcol * SZ.v tile + SZ.v bcol);

  while (!bk <^ mshared)
    invariant live bk ** live sum ** B.barrier_state (2 * !bk)
    invariant pure (v_approximates !sum (MS.__gmatmul_single 0.0R ( *. ) ( +. ) rA rB grow gcol (SZ.v !bk * SZ.v tile)))
    invariant live sa1 #(1.0R /. (tile * tile))
    invariant live sa2 #(1.0R /. (tile * tile))
    decreases (mshared - !bk)
  {
    array2_extract_tile_ro gA tile tile mrow !bk;
    let aTile = array2_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v !bk);
    assert rewrites_to aTile (array2_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v !bk));
    array2_extract_tile_ro gB tile tile !bk mcol;
    let bTile = array2_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v mcol);
    assert rewrites_to bTile (array2_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v mcol));

    let v1 = M.read aTile (brow, bcol);
    let v2 = M.read bTile (brow, bcol);

    ambig_trade_elim ();
    ambig_trade_elim ();

    even_2x !bk;

    (* Even barrier: give shared ownership, receive per-cell ownership *)
    rewrite (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
            (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
         as barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk) tid;
    rewrite barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk) tid
         as (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rin (2 * !bk) tid;

    B.barrier_wait ();

    rewrite (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rout (2 * !bk) tid
         as barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk) tid;
    rewrite barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk) tid
         as live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile);
    rewrite each (tid / tile) as v brow;
    rewrite each (tid % tile) as v bcol;

    (* Write to shmem: unfold live_cell, write, keep specific content *)
    unfold live_cell sa1 (v brow) (v bcol);
    M.write_cell sa1 (brow, bcol) v1;

    unfold live_cell sa2 (v brow) (v bcol);
    M.write_cell sa2 (brow, bcol) v2;

    (* Odd barrier: give per-cell ownership with specific content *)
    odd_2x1 !bk;
    assert (pure (odd (2 * !bk + 1)));

    rewrite each (v brow) as (tid / tile);
    rewrite each (v bcol) as (tid % tile);
    rewrite each v1 as (macc (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v !bk)) (tid / tile) (tid % tile));
    rewrite each v2 as (macc (ematrix_subtile eB tile tile (SZ.v !bk) (SZ.v mcol)) (tid / tile) (tid % tile));

    rewrite M.pts_to_cell sa1 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v !bk)) (tid/tile) (tid%tile)) **
            M.pts_to_cell sa2 (Mktuple2 #(natlt tile) #(natlt tile) (tid/tile) (tid%tile)) (macc (ematrix_subtile eB tile tile (SZ.v !bk) (SZ.v mcol)) (tid/tile) (tid%tile))
         as barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk + 1) tid;

    // fold barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk + 1) tid;

    assert pure (barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk + 1) tid
                 == (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rin (2 * !bk + 1) tid);
    rewrite barrier_p_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk + 1) tid
        as (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rin (2 * !bk + 1) tid;

    B.barrier_wait ();

    even_2x (!bk + 1);
    assert pure (2 * (!bk + 1) == 2 * !bk + 2);
    assert pure (odd (2 * !bk + 1));
    assert pure (even (2 * !bk + 2));
    assert pure (SZ.v !bk < mshared);
    assert pure ((2 * SZ.v !bk + 1) < 2 * mshared);
    assert pure ((2 * SZ.v !bk + 1) / 2 == SZ.v !bk);

    rewrite (shmem_contract tile eA eB (SZ.v bid) sa1 sa2).rout (2 * !bk + 1) tid
         as barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk + 1) tid;
    rewrite barrier_q_cell tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk + 1) tid
         as sa1 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v !bk)) **
            sa2 |-> Frac (1.0R /. (tile * tile)) (ematrix_subtile eB tile tile (SZ.v !bk) (SZ.v mcol));

    rewrite each (tid / tile) as v brow;
    rewrite each (tid % tile) as v bcol;

    let t = Kuiper.DotProd.matmul_dotprod sa1 sa2 brow bcol;
    let s = !sum;
    sum := s `add` t;

    let sub_rA = ematrix_subtile rA tile tile (SZ.v mrow) (SZ.v !bk);
    let sub_rB = ematrix_subtile rB tile tile (SZ.v !bk) (SZ.v mcol);
    MU.__matmul_single_approx_real
      (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v !bk))
      (ematrix_subtile eB tile tile (SZ.v !bk) (SZ.v mcol))
      sub_rA sub_rB
      brow bcol tile;

    let r_partial = MS.__gmatmul_single 0.0R ( *. ) ( +. ) rA rB grow gcol (SZ.v !bk * SZ.v tile);
    let r_subtile = MS.__gmatmul_single 0.0R ( *. ) ( +. ) sub_rA sub_rB brow bcol tile;

    MU.__gmatmul_single_split rA rB grow gcol (SZ.v !bk * SZ.v tile) tile sub_rA sub_rB brow bcol;
    assert (pure (
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) rA rB grow gcol (SZ.v !bk * SZ.v tile + SZ.v tile)
      == r_partial +. r_subtile));
    assert (pure ((SZ.v !bk + 1) * SZ.v tile == SZ.v !bk * SZ.v tile + SZ.v tile));

    a_add s t r_partial r_subtile;

    assert (pure (2 * (!bk + 1) == 2 * !bk + 1 + 1));

    bk := !bk +^ 1sz;
    ()
  };

  (* After the loop, vbk == mshared, so:
     s %~ __gmatmul_single ... rA rB grow gcol (mshared * tile)
        == MS.matmul_single rA rB grow gcol *)

  let v0 = M.read_cell gTile (brow, bcol);
  let v1 = comb v0 !sum;
  M.write_cell gTile (brow, bcol) v1;

  rewrite M.pts_to_cell gTile (Mktuple2 #(natlt tile) #(natlt tile) (SZ.v brow) (SZ.v bcol)) v1
       as M.pts_to_cell gTile (Mktuple2 #(natlt tile) #(natlt tile) (tid / tile) (tid % tile)) v1;

  M.lower sa1; rewrite each M.core sa1 as ar1;
  M.lower sa2; rewrite each M.core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  rewrite each gTile as array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);

  fold (kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);

  (* Fold live_c_shmem for each shmem array *)
  rewrite (exists* v. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) v)
      as  (live_c_shmem (fst sh) #(1.0R /. (tile * tile)));
  rewrite (exists* v. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) v)
      as  (live_c_shmem (fst (snd sh)) #(1.0R /. (tile * tile)));

  (* Fold live_c_shmems back for kpost *)
  fold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));
  fold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
  fold_live_c_shmems_cons sh #(1.0R /. (tile * tile));

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
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  admit();
  (* Step 1: Share gA/gB, explode gC *)
  M.share_n gA ((mrows * tile) * (mcols * tile));
  M.share_n gB ((mrows * tile) * (mcols * tile));
  array2_explode_tiled gC (SZ.v tile) (SZ.v tile);
  forevery_rw_size4 ((mrows * tile) / tile) mrows ((mcols * tile) / tile) mcols (SZ.v tile) tile (SZ.v tile) tile;

  (* Step 2: Factor gA/gB to 2D *)
  forevery_factor ((mrows * tile) * (mcols * tile)) (mrows * mcols) (SZ.v tile * SZ.v tile) (fun _ -> gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA);
  forevery_factor ((mrows * tile) * (mcols * tile)) (mrows * mcols) (SZ.v tile * SZ.v tile) (fun _ -> gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB);

  (* Step 3: Convert 4D -> 2D for gC *)
  assert pure (forall (mrow:natlt mrows) (mcol:natlt mcols). (mrow * mcols + mcol) / mcols == mrow /\ (mrow * mcols + mcol) % mcols == mcol);
  assert pure (forall (brow:natlt tile) (bcol:natlt tile). (brow * tile + bcol) / tile == brow /\ (brow * tile + bcol) % tile == bcol);
  forevery_ext_4
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol) (brow, bcol) (macc eC (mrow * tile + brow) (mcol * tile + bcol)))
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      let bid = mrow * mcols + mcol in let tid = brow * tile + bcol in
      M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile, tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  forevery_unfactor_2 (mrows * mcols) mrows mcols (SZ.v tile * SZ.v tile) (SZ.v tile) (SZ.v tile)
    (fun bid tid -> M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile, tid % tile)
      (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 4: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile, tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 5: Bridge to natlt2 and match kpre1 *)
  forevery_rw_size2 (mrows * mcols) (SZ.v (mrows `SZ.mul` mcols)) (SZ.v tile * SZ.v tile) (SZ.v (tile `SZ.mul` tile));
  forevery_ext_2
    (fun (bid : natlt (SZ.v (mrows `SZ.mul` mcols))) (tid : natlt (SZ.v (tile `SZ.mul` tile))) ->
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA ** gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB **
      M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile, tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))))
    (fun (bid : natlt2 mrows mcols) (tid : natlt2 tile tile) -> kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ();
}
#pop-options

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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

  // Bridge natlt type: natlt (tile * tile) → natlt2 tile tile
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));

  // Zip shmem fracs with kpre1 to form kpre
  forevery_zip
    (fun (tid : natlt2 tile tile) ->
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;

  ();
}

#push-options "--z3rlimit 20"
ghost
fn block_teardown
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
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
  admit();
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
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA **
      gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB **
      (exists* (v : et).
        M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile, tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB **
      (exists* (v : et).
        M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile, tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      exists* (v : et).
        M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile, tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' ((mrows * tile) * (mcols * tile)) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA);
  M.gather_n gA ((mrows * tile) * (mcols * tile));
  forevery_unfactor' ((mrows * tile) * (mcols * tile)) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB);
  M.gather_n gB ((mrows * tile) * (mcols * tile));

  (* Step 6: Collect gC cells back into matrix *)
  admit(); // TODO: array2_collect_approx_tiled needs explicit pair index ascriptions
  with eC'. assert (gC |-> eC');
  assert pure (eC' %~ MS.mmcomb comb_r rC rA rB);
  ();
}

#push-options "--z3rlimit_factor 10 --fuel 0 --ifuel 0 --split_queries no"
#restart-solver

#push-options "--z3rlimit 40"
let kpre_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
: is_send_across block_of (kpre comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid)
= magic() // solve

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
: is_send_across block_of (kpost comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid)
= magic() // solve

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
= magic() // solve

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
= magic() // solve
#pop-options

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : M.layout (mrows   * tile) (mshared * tile))
  (#lB : M.layout (mshared * tile) (mcols   * tile))
  (#lC : M.layout (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (#_ : squash (SZ.fits (M.layout_size slA)))
  (#_ : squash (SZ.fits (M.layout_size slB)))
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
  barrier_ok = magic(); // TODO: barrier_p_to_q_cell_transform needs forevery_map fix for Array2 pair indexing

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

  block_pre_sendable=block_pre_gpu_sendable comb comb_r tile slA slB gA gB gC eA eB eC fA fB;
  block_post_sendable=block_post_gpu_sendable comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;
  kpre_sendable=kpre_block_sendable comb comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost_sendable=kpost_block_sendable comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (rA  : ematrix real (m * tile) (k * tile))
  (rB  : ematrix real (k * tile) (n * tile))
  (rC  : ematrix real (m * tile) (n * tile))
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (#eC : ematrix et (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  open Kuiper.Tensor.Layout.Alg;
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile (l2_row_major _ _) (l2_row_major _ _) comb comb_r gA gB gC rA rB rC ());
}
#pop-options
