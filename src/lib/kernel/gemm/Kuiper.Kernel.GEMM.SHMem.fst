module Kuiper.Kernel.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Tensor.Tiling
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Tensor
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
open Kuiper.Chest

let live_cell
  (#et : Type0)
  (#m #n : nat)
  (#lm : layout2 m n)
  (gm : array2 et lm)
  (i : natlt m)
  (j : natlt n)
  : slprop
  = exists* v. tensor_pts_to_cell gm (idx2 i j) v

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |} (tile:valid_tile) : list shmem_desc = [
  SHArray et (tile *^ tile);
  SHArray et (tile *^ tile);
]

(* Helper: bridge between pair-indexed forall+ (from tensor_explode2)
   and 2-arg indexed forall+ (for forevery_unfactor' etc.) *)
ghost
fn explode2
  (#et : Type0) (#m #n : nat) (#l : layout2 m n)
  (a : array2 et l)
  (#f : perm)
  (#s : chest2 et m n)
  requires a |-> Frac f s
  ensures
    forall+ (r : natlt m) (c : natlt n).
      tensor_pts_to_cell a #f (idx2 r c) (acc2 s r c)
{
  tensor_explode2 a;
  forevery_ext _ (fun i -> tensor_pts_to_cell a #f (idx2 (fst i) (snd i)) (acc2 s (fst i) (snd i)));
  forevery_unflatten (fun r c -> tensor_pts_to_cell a #f (idx2 r c) (acc2 s r c));
  ()
}

ghost
fn implode2
  (#et : Type0) (#m #n : nat) (#l : layout2 m n)
  (a : array2 et l)
  (#f : perm)
  (#s : chest2 et m n)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+ (r : natlt m) (c : natlt n).
      tensor_pts_to_cell a #f (idx2 r c) (acc2 s r c)
  ensures a |-> Frac f s
{
  forevery_flatten (fun r c -> tensor_pts_to_cell a #f (idx2 r c) (acc2 s r c));
  forevery_ext _ (fun i -> tensor_pts_to_cell a #f (idx2 (fst i) (snd i)) (acc2 s (fst i) (snd i)));
  tensor_implode2 a;
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
  (eA : chest2 et (mrows * tile) (mshared * tile))
  (eB : chest2 et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
  : B.barrier_side (tile * tile)
  = fun it tid ->
    if it >= 2 * mshared then
      emp
    else if even it then
      (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
      (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
    else
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eA tile tile mrow (it / 2)) (tid/tile) (tid%tile)) **
      tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eB tile tile (it / 2) mcol) (tid/tile) (tid%tile))

(* Fold the even-step shared-ownership form into barrier_p_cell.
   The runtime [if]s make the [it >= 2*mshared] and [even it] guards in
   barrier_p_cell reducible so the [match] collapses to the even branch;
   the impossible branches are discharged via [unreachable]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
ghost
fn fold_barrier_p_even
  (#et : Type0)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 et (mrows * tile) (mshared * tile))
  (eB : chest2 et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
  (it : nat)
  (tid : natlt (tile * tile))
  requires
    (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
    (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x) **
    pure (even it /\ it < 2 * mshared)
  ensures
    barrier_p_cell tile eA eB bid sa1 sa2 it tid
{
  if (it >= 2 * mshared) {
    unreachable ();
  } else {
    let ev = even it;
    if ev {
      rewrite (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
              (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
           as barrier_p_cell tile eA eB bid sa1 sa2 it tid;
    } else {
      unreachable ();
    }
  }
}
#pop-options

let barrier_q_cell
  (#et : Type0) {| scalar et |}
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 et (mrows * tile) (mshared * tile))
  (eB : chest2 et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
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
  (eA : chest2 et (mrows * tile) (mshared * tile))
  (eB : chest2 et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
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
  (eA : chest2 et (mrows * tile) (mshared * tile))
  (eB : chest2 et (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 et slA)
  (sa2 : array2 et slB)
  (#_ : squash (SZ.fits (slA.ulen)))
  (#_ : squash (SZ.fits (slB.ulen)))
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
          (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
          (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x))
        fn tid {
          rewrite barrier_p_cell tile eA eB bid sa1 sa2 it tid
               as (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
                  (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x);
        };
      forevery_unzip _ _;
      tensor_gather_n_underspec sa1 (tile * tile);
      tensor_gather_n_underspec sa2 (tile * tile);
      with em1. assert (pts_to sa1 #1.0R em1);
      with em2. assert (pts_to sa2 #1.0R em2);
      explode2 sa1;
      forevery_unfactor' (tile * tile) tile tile
        (fun r c -> tensor_pts_to_cell sa1 (idx2 r c) (acc2 em1 r c));
      explode2 sa2;
      forevery_unfactor' (tile * tile) tile tile
        (fun r c -> tensor_pts_to_cell sa2 (idx2 r c) (acc2 em2 r c));
      forevery_map
        (fun (tid : natlt (tile * tile)) -> tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 em1 (tid/tile) (tid%tile)))
        (fun (tid : natlt (tile * tile)) -> live_cell sa1 (tid/tile) (tid%tile))
        fn tid { fold live_cell };
      forevery_map
        (fun (tid : natlt (tile * tile)) -> tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 em2 (tid/tile) (tid%tile)))
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
          tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eA tile tile mrow (it/2)) (tid/tile) (tid%tile)) **
          tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eB tile tile (it/2) mcol) (tid/tile) (tid%tile)))
        fn tid {
          rewrite barrier_p_cell tile eA eB bid sa1 sa2 it tid
               as tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eA tile tile mrow (it/2)) (tid/tile) (tid%tile)) **
                  tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eB tile tile (it/2) mcol) (tid/tile) (tid%tile));
        };
      forevery_unzip _ _;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> tensor_pts_to_cell sa1 (idx2 r c) (acc2 (ematrix_subtile eA tile tile mrow (it/2)) r c));
      implode2 sa1;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> tensor_pts_to_cell sa2 (idx2 r c) (acc2 (ematrix_subtile eB tile tile (it/2) mcol) r c));
      implode2 sa2;
      tensor_share_n sa1 (tile * tile);
      tensor_share_n sa2 (tile * tile);
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
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
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
  tensor_pts_to_cell
    (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
    #1.0R
    (idx2 (tid / tile) (tid % tile)) (acc2 eC grow gcol)

(* Functional postcondition: the cell contains a value approximating
   MS.gemm_single over external real matrices rA, rB, rC *)
unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
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
    tensor_pts_to_cell
      (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      #1.0R
      (idx2 (tid / tile) (tid % tile)) v **
    pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol))

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
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
  (slA slB : full_layout2 tile tile) // shmem layouts
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
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
  (slA slB : full_layout2 tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#eA : chest2 et (mrows * tile)   (mshared * tile))
  (#eB : chest2 et (mshared * tile) (mcols * tile))
  (#eC : chest2 et (mrows * tile)   (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
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
    B.barrier_tok (shmem_contract tile eA eB bid (from_array slA (fst sh)) (from_array slB (fst (snd sh)))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (shmem_contract tile eA eB bid (from_array slA (fst sh)) (from_array slB (fst (snd sh)))) **
    B.barrier_state (2 * mshared)
{
  (* Unfold live_c_shmems to get raw gpu_pts_to_array *)
  unfold_live_c_shmems_cons sh #(1.0R /. (tile * tile));
  unfold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
  unfold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));

  let (ar1, (ar2, _)) = sh;

  rewrite (live_c_shmem ar1 #(1.0R /. (tile * tile)))
      as  (exists* v. pts_to ar1 #(1.0R /. (tile * tile)) v);
  rewrite (live_c_shmem ar2 #(1.0R /. (tile * tile)))
      as  (exists* v. pts_to ar2 #(1.0R /. (tile * tile)) v);

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  tensor_abs' slA ar1;
  let sa1 = from_array slA ar1;
  rewrite each from_array slA ar1 as sa1;

  tensor_abs' slB ar2;
  let sa2 = from_array slB ar2;
  rewrite each from_array slB ar2 as sa2;

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

    let v1 = tensor_read aTile ((brow <: szlt _), ((bcol <: szlt _), ()));
    let v2 = tensor_read bTile ((brow <: szlt _), ((bcol <: szlt _), ()));

    ambig_trade_elim ();
    ambig_trade_elim ();

    even_2x !bk;
    assert (pure (2 * SZ.v !bk < 2 * SZ.v mshared));
    assert (pure (Kuiper.Math.even (2 * SZ.v !bk)));

    (* Even barrier: give shared ownership, receive per-cell ownership *)
    fold_barrier_p_even tile eA eB (SZ.v bid) sa1 sa2 (2 * !bk) tid;
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
    tensor_write_cell sa1 ((brow <: szlt _), ((bcol <: szlt _), ())) v1;

    unfold live_cell sa2 (v brow) (v bcol);
    tensor_write_cell sa2 ((brow <: szlt _), ((bcol <: szlt _), ())) v2;

    (* Odd barrier: give per-cell ownership with specific content *)
    odd_2x1 !bk;
    assert (pure (odd (2 * !bk + 1)));

    rewrite each (v brow) as (tid / tile);
    rewrite each (v bcol) as (tid % tile);
    rewrite each v1 as (acc2 (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v !bk)) (tid / tile) (tid % tile));
    rewrite each v2 as (acc2 (ematrix_subtile eB tile tile (SZ.v !bk) (SZ.v mcol)) (tid / tile) (tid % tile));

    rewrite tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eA tile tile (SZ.v mrow) (SZ.v !bk)) (tid/tile) (tid%tile)) **
            tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (ematrix_subtile eB tile tile (SZ.v !bk) (SZ.v mcol)) (tid/tile) (tid%tile))
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

    assert (pure (2 * (!bk + 1) == 2 * !bk + 1 + 1));

    bk := !bk +^ 1sz;
    ()
  };

  (* After the loop, vbk == mshared, so:
     s %~ __gmatmul_single ... rA rB grow gcol (mshared * tile)
        == MS.matmul_single rA rB grow gcol *)

  let v0 = tensor_read_cell gTile ((brow <: szlt _), ((bcol <: szlt _), ()));
  let v1 = comb v0 !sum;
  tensor_write_cell gTile ((brow <: szlt _), ((bcol <: szlt _), ())) v1;

  rewrite tensor_pts_to_cell gTile (idx2 (SZ.v brow) (SZ.v bcol)) v1
       as tensor_pts_to_cell gTile (idx2 (tid / tile) (tid % tile)) v1;

  tensor_concr sa1; rewrite each core sa1 as ar1;
  tensor_concr sa2; rewrite each core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  rewrite each gTile as array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);

  fold (kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);

  (* Fold live_c_shmem for each shmem array *)
  rewrite (exists* v. pts_to (fst sh) #(1.0R /. (tile * tile)) v)
      as  (live_c_shmem (fst sh) #(1.0R /. (tile * tile)));
  rewrite (exists* v. pts_to (fst (snd sh)) #(1.0R /. (tile * tile)) v)
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
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#fA #fB : perm)
  (#eA #eB : chest2 _ _ _)
  (#eC : chest2 et (mrows * tile) (mcols * tile))
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
  ();
  (* Step 1: Share gA/gB, explode gC *)
  tensor_share_n gA ((mrows * tile) * (mcols * tile));
  tensor_share_n gB ((mrows * tile) * (mcols * tile));
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
      tensor_pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol) (idx2 (brow <: natlt tile) (bcol <: natlt tile)) (acc2 eC (mrow * tile + brow) (mcol * tile + bcol)))
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      let bid = mrow * mcols + mcol in let tid = brow * tile + bcol in
      tensor_pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile))
        (acc2 eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  forevery_unfactor_2 (mrows * mcols) mrows mcols (SZ.v tile * SZ.v tile) (SZ.v tile) (SZ.v tile)
    (fun bid tid -> tensor_pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile))
      (acc2 eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 4: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      tensor_pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile))
        (acc2 eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 5: Bridge to natlt2 and match kpre1 *)
  forevery_rw_size2 (mrows * mcols) (SZ.v (mrows `SZ.mul` mcols)) (SZ.v tile * SZ.v tile) (SZ.v (tile `SZ.mul` tile));
  forevery_ext_2
    (fun (bid : natlt (SZ.v (mrows `SZ.mul` mcols))) (tid : natlt (SZ.v (tile `SZ.mul` tile))) ->
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA ** gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB **
      tensor_pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile))
        (acc2 eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))))
    (fun (bid : natlt2 mrows mcols) (tid : natlt2 tile tile) -> kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ();
}
#pop-options

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#fA #fB : perm)
  (#eA : chest2 et (mrows * tile) (mshared * tile))
  (#eB : chest2 et (mshared * tile) (mcols * tile))
  (#eC : chest2 et (mrows * tile) (mcols * tile))
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
  (slA slB : full_layout2 tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#fA #fB : perm)
  (#eA : chest2 et (mrows * tile) (mshared * tile))
  (#eB : chest2 et (mshared * tile) (mcols * tile))
  (#eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
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
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#fA #fB : perm)
  (#eA : chest2 et (mrows * tile) (mshared * tile))
  (#eB : chest2 et (mshared * tile) (mcols * tile))
  (#eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
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
    (exists* (eC' : chest2 et (mrows * tile) (mcols * tile)).
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
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA **
      gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB **
      (exists* (v : et).
        tensor_pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile)) v **
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
        tensor_pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile)) v **
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
        tensor_pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (idx2 (tid / tile <: natlt tile) (tid % tile <: natlt tile)) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' ((mrows * tile) * (mcols * tile)) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. ((mrows * tile) * (mcols * tile))) eA);
  tensor_gather_n gA ((mrows * tile) * (mcols * tile));
  forevery_unfactor' ((mrows * tile) * (mcols * tile)) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. ((mrows * tile) * (mcols * tile))) eB);
  tensor_gather_n gB ((mrows * tile) * (mcols * tile));

  (* Step 6: Collect gC cells back into matrix *)
  let vf = Kuiper.Tensor.Tiling.CollectApprox.array2_collect_approx_tiled gC (SZ.v tile) (SZ.v tile) mrows mcols
    (fun row col v -> v %~ MS.gemm_single comb_r rA rB rC row col);
  with eC'. assert (gC |-> eC');
  assert pure (eC' %~ MS.mmcomb comb_r rC rA rB);
  ();
}

#push-options "--z3rlimit_factor 10 --fuel 0 --ifuel 0 --split_queries no"

#push-options "--z3rlimit 40"
let kpre_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
: is_send_across block_of (kpre comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid)
= solve

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
: is_send_across block_of (kpost comb comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid)
= solve

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
= solve

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (eA eB : chest2 _ _ _)
  (eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
= solve
#pop-options

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : layout2 (mrows   * tile) (mshared * tile))
  (#lB : layout2 (mshared * tile) (mcols   * tile))
  (#lC : layout2 (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (#fA #fB : perm)
  (#eA : chest2 et (mrows * tile) (mshared * tile))
  (#eB : chest2 et (mshared * tile) (mcols * tile))
  (#eC : chest2 et (mrows * tile) (mcols * tile))
  (rA : chest2 real (mrows   * tile) (mshared * tile))
  (rB : chest2 real (mshared * tile) (mcols   * tile))
  (rC : chest2 real (mrows   * tile) (mcols   * tile))
  (#_ : squash (SZ.fits (slA.ulen)))
  (#_ : squash (SZ.fits (slB.ulen)))
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads
               /\ eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : chest2 et (mrows * tile) (mcols * tile)).
          gC |-> eC' **
          pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  barrier_contract = (fun _bid ptrs -> shmem_contract tile eA eB _bid (from_array slA (fst ptrs)) (from_array slB (fst (snd ptrs))));
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun _bid ptrs -> barrier_p_to_q_cell_transform tile eA eB _bid (from_array slA (fst ptrs)) (from_array slB (fst (snd ptrs))));

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
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  (#lC : layout2 (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (rA  : chest2 real (m * tile) (k * tile))
  (rB  : chest2 real (k * tile) (n * tile))
  (rC  : chest2 real (m * tile) (n * tile))
  (#eA : chest2 et (m * tile) (k * tile))
  (#eB : chest2 et (k * tile) (n * tile))
  (#eC : chest2 et (m * tile) (n * tile))
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
    (exists* (eC' : chest2 et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  open Kuiper.Tensor.Layout.Alg;
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile (l2_row_major _ _) (l2_row_major _ _) comb comb_r gA gB gC rA rB rC ());
}
#pop-options
