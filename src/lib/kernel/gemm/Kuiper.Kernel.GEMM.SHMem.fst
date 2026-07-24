module Kuiper.Kernel.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Tensor.Tiling
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Tensor
open Kuiper.Tensor.Layout.Slice
open Pulse.Lib.Trade
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module C = Kuiper.Matrix.Casts
open Kuiper.Bijection
open Kuiper.Chest
module Chest = Kuiper.Chest

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
  (#ta #tb #tacc : Type0)
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 ta (mrows * tile) (mshared * tile))
  (eB : chest2 tb (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 tacc slA)
  (sa2 : array2 tacc slB)
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
      tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it / 2))) (tid/tile) (tid%tile)) **
      tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (Chest.chest_map mapB (ematrix_subtile eB tile tile (it / 2) mcol)) (tid/tile) (tid%tile))

(* Fold the even-step shared-ownership form into barrier_p_cell.
   The runtime [if]s make the [it >= 2*mshared] and [even it] guards in
   barrier_p_cell reducible so the [match] collapses to the even branch;
   the impossible branches are discharged via [unreachable]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
ghost
fn fold_barrier_p_even
  (#ta #tb #tacc : Type0)
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 ta (mrows * tile) (mshared * tile))
  (eB : chest2 tb (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 tacc slA)
  (sa2 : array2 tacc slB)
  (it : nat)
  (tid : natlt (tile * tile))
  requires
    (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
    (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x) **
    pure (even it /\ it < 2 * mshared)
  ensures
    barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid
{
  if (it >= 2 * mshared) {
    unreachable ();
  } else {
    let ev = even it;
    if ev {
      rewrite (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
              (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
           as barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid;
    } else {
      unreachable ();
    }
  }
}
#pop-options

let barrier_q_cell
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 ta (mrows * tile) (mshared * tile))
  (eB : chest2 tb (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 tacc slA)
  (sa2 : array2 tacc slB)
  : B.barrier_side (tile * tile)
  = fun it tid ->
    if it >= 2 * mshared then
      emp
    else if even it then
      live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile)
    else
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      sa1 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it / 2))) **
      sa2 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapB (ematrix_subtile eB tile tile (it / 2) mcol))

let shmem_contract
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 ta (mrows * tile) (mshared * tile))
  (eB : chest2 tb (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 tacc slA)
  (sa2 : array2 tacc slB)
  : B.contract (tile * tile) = {
    rin  = barrier_p_cell mapA mapB tile eA eB bid sa1 sa2;
    rout = barrier_q_cell mapA mapB tile eA eB bid sa1 sa2;
  }

#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
ghost
fn barrier_p_to_q_cell_transform
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (tile : valid_tile)
  (#mrows #mshared #mcols : pos)
  (eA : chest2 ta (mrows * tile) (mshared * tile))
  (eB : chest2 tb (mshared * tile) (mcols * tile))
  (bid : natlt (mrows * mcols))
  (#slA #slB : full_layout2 tile tile)
  (sa1 : array2 tacc slA)
  (sa2 : array2 tacc slB)
  (#_ : squash (SZ.fits (slA.ulen)))
  (#_ : squash (SZ.fits (slB.ulen)))
  (it : nat)
  requires
    forall+ (tid : natlt (tile * tile)).
      barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid
  ensures
    forall+ (tid : natlt (tile * tile)).
      barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid
{
  if (it >= 2 * mshared) {
    forevery_map
      (fun (tid : natlt (tile * tile)) -> barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid)
      (fun (tid : natlt (tile * tile)) -> barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid)
      fn tid {
        rewrite barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid as emp;
        rewrite emp as barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid;
      };
  } else {
    let ev = even it;
    if ev {
      assert pure (it < 2 * mshared);
      assert pure (even it);
      forevery_map
        (fun (tid : natlt (tile * tile)) -> barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid)
        (fun (tid : natlt (tile * tile)) ->
          (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
          (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x))
        fn tid {
          rewrite barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid
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
        (fun (tid : natlt (tile * tile)) -> barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid)
        fn tid {
          rewrite live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile)
               as barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid;
        };
    } else {
      assert pure (it < 2 * mshared);
      assert pure (odd it);
      let mrow = bid / mcols;
      let mcol = bid % mcols;
      forevery_map
        (fun (tid : natlt (tile * tile)) -> barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid)
        (fun (tid : natlt (tile * tile)) ->
          tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it/2))) (tid/tile) (tid%tile)) **
          tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (Chest.chest_map mapB (ematrix_subtile eB tile tile (it/2) mcol)) (tid/tile) (tid%tile)))
        fn tid {
          rewrite barrier_p_cell mapA mapB tile eA eB bid sa1 sa2 it tid
               as tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (acc2 (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it/2))) (tid/tile) (tid%tile)) **
                  tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (acc2 (Chest.chest_map mapB (ematrix_subtile eB tile tile (it/2) mcol)) (tid/tile) (tid%tile));
        };
      forevery_unzip _ _;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> tensor_pts_to_cell sa1 (idx2 r c) (acc2 (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it/2))) r c));
      implode2 sa1;
      forevery_factor' (tile * tile) tile tile
        (fun r c -> tensor_pts_to_cell sa2 (idx2 r c) (acc2 (Chest.chest_map mapB (ematrix_subtile eB tile tile (it/2) mcol)) r c));
      implode2 sa2;
      tensor_share_n sa1 (tile * tile);
      tensor_share_n sa2 (tile * tile);
      forevery_zip
        (fun (_ : natlt (tile * tile)) -> sa1 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it/2)))) _;
      forevery_map
        (fun (tid : natlt (tile * tile)) ->
          sa1 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it/2))) **
          sa2 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapB (ematrix_subtile eB tile tile (it/2) mcol)))
        (fun (tid : natlt (tile * tile)) -> barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid)
        fn tid {
          rewrite
            sa1 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapA (ematrix_subtile eA tile tile mrow (it/2))) **
            sa2 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapB (ematrix_subtile eB tile tile (it/2) mcol))
          as
            barrier_q_cell mapA mapB tile eA eB bid sa1 sa2 it tid;
        };
    }
  }
}
#pop-options

(* ═══ batched index bijection (rank-3 output, tiled) ═══════════════════════════
   Maps the abstract rank-3 output index [(page,(grow,(gcol,())))] to the flat
   [(bid, tid)] pair, where the block id is PAGE-MINOR
     bid = rest * batch + page,   rest = mrow * mcols + mcol
   so that [gg (bid, tid) = (page, (grow, (gcol, ())))] with
     page = bid % batch,
     grow = ((bid / batch) / mcols) * tile + tid / tile,
     gcol = ((bid / batch) % mcols) * tile + tid % tile.
   Built from combinators so the arithmetic round-trips come for free; only the
   two tuple-shuffles carry raw (definitional) proofs. *)

(* Identity bijection witnessing commutativity of the block-count product. *)
let sbij_comm_size (a b : nat) : (natlt (a * b) =~ natlt (b * a)) =
  {
    ff = (fun (x : natlt (a * b)) -> (x <: natlt (b * a)));
    gg = (fun (x : natlt (b * a)) -> (x <: natlt (a * b)));
    ff_gg = (fun x -> ());
    gg_ff = (fun x -> ());
  }

(* Shuffle: pull the page index out to the tail. *)
let sabs_shuffle (batch gr gc : nat)
  : (abs (batch @| gr @| gc @| INil) =~ ((natlt gr & natlt gc) & natlt batch))
  = {
      ff = (fun (pg, (r, (c, ()))) -> ((r, c), pg));
      gg = (fun ((r, c), pg) -> (pg, (r, (c, ()))));
      ff_gg = (fun ((r, c), pg) -> ());
      gg_ff = (fun (pg, (r, (c, ()))) -> ());
    }

(* Shuffle: regroup split (row,col)+page into ((mrow,mcol),page) + (brow,bcol). *)
let sabs_regroup (mrows mcols tile batch : nat)
  : (((natlt mrows & natlt tile) & (natlt mcols & natlt tile)) & natlt batch
     =~ ((natlt mrows & natlt mcols) & natlt batch) & (natlt tile & natlt tile))
  = {
      ff = (fun (((mr, br), (mc, bc)), pg) -> (((mr, mc), pg), (br, bc)));
      gg = (fun (((mr, mc), pg), (br, bc)) -> (((mr, br), (mc, bc)), pg));
      ff_gg = (fun (((mr, mc), pg), (br, bc)) -> ());
      gg_ff = (fun (((mr, br), (mc, bc)), pg) -> ());
    }

(* Flatten the block components in page-minor order. *)
let sflat_block (mrows mcols batch : nat)
  : ((natlt mrows & natlt mcols) & natlt batch =~ natlt (batch * (mrows * mcols)))
  = bij_comp
      (bij_prod (bij_nat_prod #mrows #mcols) (bij_self (natlt batch)))
      (bij_comp (bij_nat_prod #(mrows * mcols) #batch)
                (sbij_comm_size (mrows * mcols) batch))

let sbtile_idx_bij (batch mrows mcols tile : nat)
  : (abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)
     =~ natlt (batch * (mrows * mcols)) & natlt (tile * tile))
  = bij_comp (sabs_shuffle batch (mrows * tile) (mcols * tile))
      (bij_comp
         (bij_prod
            (bij_prod (bij_sym (bij_nat_prod #mrows #tile))
                      (bij_sym (bij_nat_prod #mcols #tile)))
            (bij_self (natlt batch)))
         (bij_comp (sabs_regroup mrows mcols tile batch)
            (bij_prod (sflat_block mrows mcols batch)
                      (bij_nat_prod #tile #tile))))

(* The direct page-minor arithmetic cell index for the output tensor.
   [prod_ff] supplies the [< mrows*tile] / [< mcols*tile] bounds for free. *)
unfold
let sbtile_cell_idx (batch mrows mcols tile : nat)
  (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile))
  : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)
  = let page = bid % batch in
    let rest = bid / batch in
    let mrow = rest / mcols in
    let mcol = rest % mcols in
    let brow = tid / tile in
    let bcol = tid % tile in
    ((page <: natlt batch),
      ((prod_ff mrows tile ((mrow <: natlt mrows), (brow <: natlt tile))),
        ((prod_ff mcols tile ((mcol <: natlt mcols), (bcol <: natlt tile))), ())))

(* Slicing preserves the approximation relation (cellwise). *)
let chest_slice_approx
  (#et : Type) {| scalar et, real_like et |}
  (#r : nat) (#d : shape r)
  (i : natlt r) (j : natlt (d @! i))
  (e : chest d et) (rr : chest d real)
  : Lemma (requires e %~ rr)
          (ensures chest_slice i j e %~ chest_slice i j rr)
  = introduce forall (idx : abs (Kuiper.Shape.modulo_i i d)).
      Chest.acc (chest_slice i j e) idx %~ Chest.acc (chest_slice i j rr) idx
    with ()

(* The value at the direct arithmetic cell equals the acc2 of the page slice at grow/gcol. *)
#push-options "--split_queries always"
let acc_bridge
  (#tc : Type0)
  (batch mrows mcols tile : nat)
  (e : chest3 tc batch (mrows * tile) (mcols * tile))
  (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile))
  : Lemma (
      Chest.acc e (sbtile_cell_idx batch mrows mcols tile bid tid)
        == acc2 (slice_page e (bid % batch))
             (((bid / batch) / mcols) * tile + tid / tile)
             (((bid / batch) % mcols) * tile + tid % tile))
  = ()
#pop-options

(* [up] of an explicit rank-3 concrete cell index reduces componentwise. *)
#push-options "--fuel 4 --ifuel 4"
let up3_lemma (#b #r #cc : nat) (p : szlt b) (g : szlt r) (co : szlt cc)
  : Lemma (up ((p, (g, (co, ()))) <: conc (b @| r @| cc @| INil))
             == ((SZ.v p <: natlt b), ((SZ.v g <: natlt r), ((SZ.v co <: natlt cc), ()))))
  = ()
#pop-options

(* The full block/thread bijection decodes to the direct arithmetic cell index. *)
let sbtile_gg_full (batch mrows mcols tile : nat)
  (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile))
  : Lemma ((sbtile_idx_bij batch mrows mcols tile).gg (bid, tid)
             == sbtile_cell_idx batch mrows mcols tile bid tid)
  = assert_norm ((sbtile_idx_bij batch mrows mcols tile).gg (bid, tid)
                   == sbtile_cell_idx batch mrows mcols tile bid tid)

let sbtile_gg_all (batch mrows mcols tile : nat)
  : Lemma (forall (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)).
             (sbtile_idx_bij batch mrows mcols tile).gg (bid, tid)
               == sbtile_cell_idx batch mrows mcols tile bid tid)
  = introduce
      forall (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)).
        (sbtile_idx_bij batch mrows mcols tile).gg (bid, tid)
          == sbtile_cell_idx batch mrows mcols tile bid tid
      with sbtile_gg_full batch mrows mcols tile bid tid

(* A cell of the batched combined spec equals the per-page rank-2 [ggemm_single]
   cell.  Reduces the rank-3 [gbmmcomb] obligation cellwise/pagewise. *)
let bmmcomb_cell_shmem
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#batch #bm #bs #bn : nat)
  (rA : chest3 real batch bm bs)
  (rB : chest3 real batch bs bn)
  (rC : chest3 real batch bm bn)
  (page : natlt batch) (row : natlt bm) (col : natlt bn)
  : Lemma
      (Chest.acc (MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB) (page, (row, (col, ())))
        == MS.ggemm_single mapA_r mapB_r comb_r
             (slice_page rA page) (slice_page rB page) (slice_page rC page) row col)
  = ()

let bmmcomb_all_shmem
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#batch #bm #bs #bn : nat)
  (rA : chest3 real batch bm bs)
  (rB : chest3 real batch bs bn)
  (rC : chest3 real batch bm bn)
  : Lemma
      (forall (page : natlt batch) (row : natlt bm) (col : natlt bn).
        Chest.acc (MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB) (page, (row, (col, ())))
          == MS.ggemm_single mapA_r mapB_r comb_r
               (slice_page rA page) (slice_page rB page) (slice_page rC page) row col)
  = introduce
      forall (page : natlt batch) (row : natlt bm) (col : natlt bn).
        Chest.acc (MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB) (page, (row, (col, ())))
          == MS.ggemm_single mapA_r mapB_r comb_r
               (slice_page rA page) (slice_page rB page) (slice_page rC page) row col
      with bmmcomb_cell_shmem mapA_r mapB_r comb_r rA rB rC page row col


(* ══════════════════════════════════════════════════════════════════════════
   BATCHED (rank-3) KERNEL

   The batched kernel is the ONLY real kernel description; the rank-2 entry
   below is derived from it at [batch = 1].  Each block fixes a page (batch
   index), slices the rank-3 operands down to their rank-2 page views, and
   reuses the exact same barrier + shared-memory protocol as the rank-2 body.
   The block grid is PAGE-MINOR: bid = rest * batch + page.
   ══════════════════════════════════════════════════════════════════════════ *)

(* without shmem ownership *)
unfold
let bkpre1
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (batch * (mrows * mcols)))
  (tid : natlt (tile * tile))
  : slprop
  =
  let cidx = sbtile_cell_idx batch mrows mcols tile bid tid in
  (gA |-> Frac (fA /. ((batch * (mrows * mcols)) * (tile * tile))) eA) **
  (gB |-> Frac (fB /. ((batch * (mrows * mcols)) * (tile * tile))) eB) **
  tensor_pts_to_cell gC cidx (Chest.acc eC cidx)

unfold
let bkpost1
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (batch * (mrows * mcols)))
  (tid : natlt (tile * tile))
  : slprop
  =
  let page = bid % batch in
  let rest = bid / batch in
  let mrow = rest / mcols in
  let mcol = rest % mcols in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  let cidx = sbtile_cell_idx batch mrows mcols tile bid tid in
  let rA_p : chest2 real (mrows * tile) (mshared * tile) = chest_slice 0 page rA in
  let rB_p : chest2 real (mshared * tile) (mcols * tile) = chest_slice 0 page rB in
  let rC_p : chest2 real (mrows * tile) (mcols * tile) = chest_slice 0 page rC in
  (gA |-> Frac (fA /. ((batch * (mrows * mcols)) * (tile * tile))) eA) **
  (gB |-> Frac (fB /. ((batch * (mrows * mcols)) * (tile * tile))) eB) **
  (exists* (v : tc).
    tensor_pts_to_cell gC cidx v **
    pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
            rA_p rB_p rC_p
            grow gcol))

unfold
let bkpre
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc tacc tile))
  (bid : natlt (batch * (mrows * mcols)))
  (tid : natlt (tile * tile))
  : slprop
  =
  bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid **
  live_c_shmems sh #(1.0R /. (tile * tile))

unfold
let bkpost
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc tacc tile))
  (bid : natlt (batch * (mrows * mcols)))
  (tid : natlt (tile * tile))
  : slprop
  =
  bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
  live_c_shmems sh #(1.0R /. (tile * tile))

(* ─── batched thread function (page-batched barrier GEMM) ──────────────────── *)
#push-options "--z3rlimit 200 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn bkf
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (#eA : chest3 ta batch (mrows * tile)   (mshared * tile))
  (#eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (#eC : chest3 tc batch (mrows * tile)   (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (#_sq : squash (eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  (#_sq2 : squash (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc tacc tile))
  (bid : szlt (batch * (mrows * mcols)))
  (tid : szlt (tile  * tile))
  ()
  norewrite
  requires
    gpu **
    bkpre mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (batch * (mrows * mcols)) bid **
    B.barrier_tok (shmem_contract mapA mapB tile
                     (chest_slice 0 (SZ.v bid % batch) eA <: chest2 ta (mrows * tile) (mshared * tile))
                     (chest_slice 0 (SZ.v bid % batch) eB <: chest2 tb (mshared * tile) (mcols * tile))
                     (SZ.v bid / batch)
                     (from_array slA (fst sh)) (from_array slB (fst (snd sh)))) **
    B.barrier_state 0
  ensures
    gpu **
    bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (batch * (mrows * mcols)) bid **
    B.barrier_tok (shmem_contract mapA mapB tile
                     (chest_slice 0 (SZ.v bid % batch) eA <: chest2 ta (mrows * tile) (mshared * tile))
                     (chest_slice 0 (SZ.v bid % batch) eB <: chest2 tb (mshared * tile) (mcols * tile))
                     (SZ.v bid / batch)
                     (from_array slA (fst sh)) (from_array slB (fst (snd sh)))) **
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

  (* Decode the page-minor block/thread indices. *)
  let page : szlt batch = bid %^ batch;
  let rest = bid /^ batch;
  let mrow, mcol = s_divmod mcols rest;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (SZ.v page == SZ.v bid % batch));
  assert (pure (SZ.v rest == SZ.v bid / batch));
  assert (pure (SZ.v mrow == (SZ.v bid / batch) / mcols));
  assert (pure (SZ.v mcol == (SZ.v bid / batch) % mcols));
  assert (pure (SZ.v brow == tid / tile));
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  (* Ascribed page-slice views (fix pos-implicit inference for shmem_contract). *)
  let eA_p : chest2 ta (mrows * tile) (mshared * tile) = chest_slice 0 (SZ.v page) eA;
  let eB_p : chest2 tb (mshared * tile) (mcols * tile) = chest_slice 0 (SZ.v page) eB;
  let rA_p : chest2 real (mrows * tile) (mshared * tile) = chest_slice 0 (SZ.v page) rA;
  let rB_p : chest2 real (mshared * tile) (mcols * tile) = chest_slice 0 (SZ.v page) rB;
  let rC_p : chest2 real (mrows * tile) (mcols * tile) = chest_slice 0 (SZ.v page) rC;

  (* Rewrite the barrier token into decoded [page]/[rest] form. *)
  rewrite each (SZ.v bid % batch) as (SZ.v page);
  rewrite each (SZ.v bid / batch) as (SZ.v rest);
  rewrite each (chest_slice 0 (SZ.v page) eA <: chest2 ta (mrows * tile) (mshared * tile)) as eA_p;
  rewrite each (chest_slice 0 (SZ.v page) eB <: chest2 tb (mshared * tile) (mcols * tile)) as eB_p;

  rewrite each (tid / tile) as v brow;
  rewrite each (tid % tile) as v bcol;

  (* Slice out the [page]-th rank-2 page views of A and B (read-only). *)
  tensor_extract_slice_ro gA 0 (SZ.v page);
  tensor_extract_slice_ro gB 0 (SZ.v page);
  let gA_p = sliceof gA 0 (SZ.v page);
  rewrite each sliceof gA 0 (SZ.v page) as gA_p;
  let gB_p = sliceof gB 0 (SZ.v page);
  rewrite each sliceof gB 0 (SZ.v page) as gB_p;

  (* Per-page real operands still approximate their element counterparts. *)
  chest_slice_approx 0 (SZ.v page) eA rA;
  chest_slice_approx 0 (SZ.v page) eB rB;
  chest_slice_approx 0 (SZ.v page) eC rC;

  let mut sum : tacc = zero;
  let mut bk  : szle mshared = 0sz;

  let grow : erased (natlt (mrows * tile)) = hide (SZ.v mrow * SZ.v tile + SZ.v brow);
  let gcol : erased (natlt (mcols * tile)) = hide (SZ.v mcol * SZ.v tile + SZ.v bcol);

  while (!bk <^ mshared)
    invariant live bk ** live sum ** B.barrier_state (2 * !bk)
    invariant pure (v_approximates !sum (MS.__gmatmul_single 0.0R ( *. ) ( +. ) (Chest.chest_map mapA_r (rA_p)) (Chest.chest_map mapB_r (rB_p)) grow gcol (SZ.v !bk * SZ.v tile)))
    invariant live sa1 #(1.0R /. (tile * tile))
    invariant live sa2 #(1.0R /. (tile * tile))
    decreases (mshared - !bk)
  {
    array2_extract_tile_ro #ta #(mrows * tile) #(mshared * tile) gA_p tile tile mrow !bk;
    let aTile = array2_subtile #ta #(mrows * tile) #(mshared * tile) gA_p (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v !bk);
    assert rewrites_to aTile (array2_subtile #ta #(mrows * tile) #(mshared * tile) gA_p (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v !bk));
    array2_extract_tile_ro #tb #(mshared * tile) #(mcols * tile) gB_p tile tile !bk mcol;
    let bTile = array2_subtile #tb #(mshared * tile) #(mcols * tile) gB_p (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v mcol);
    assert rewrites_to bTile (array2_subtile #tb #(mshared * tile) #(mcols * tile) gB_p (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v mcol));

    let v1 = tensor_read aTile ((brow <: szlt _), ((bcol <: szlt _), ()));
    let v2 = tensor_read bTile ((brow <: szlt _), ((bcol <: szlt _), ()));

    ambig_trade_elim ();
    ambig_trade_elim ();

    even_2x !bk;
    assert (pure (2 * SZ.v !bk < 2 * SZ.v mshared));
    assert (pure (Kuiper.Math.even (2 * SZ.v !bk)));

    (* Even barrier: give shared ownership, receive per-cell ownership *)
    fold_barrier_p_even mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk) tid;
    rewrite barrier_p_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk) tid
         as (shmem_contract mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2).rin (2 * !bk) tid;

    B.barrier_wait ();

    rewrite (shmem_contract mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2).rout (2 * !bk) tid
         as barrier_q_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk) tid;
    rewrite barrier_q_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk) tid
         as live_cell sa1 (tid/tile) (tid%tile) ** live_cell sa2 (tid/tile) (tid%tile);
    rewrite each (tid / tile) as v brow;
    rewrite each (tid % tile) as v bcol;

    (* Write to shmem: unfold live_cell, write mapped value, keep specific content *)
    unfold live_cell sa1 (v brow) (v bcol);
    tensor_write_cell sa1 ((brow <: szlt _), ((bcol <: szlt _), ())) (mapA v1);

    unfold live_cell sa2 (v brow) (v bcol);
    tensor_write_cell sa2 ((brow <: szlt _), ((bcol <: szlt _), ())) (mapB v2);

    (* Odd barrier: give per-cell ownership with specific content *)
    odd_2x1 !bk;
    assert (pure (odd (2 * !bk + 1)));

    rewrite each (v brow) as (tid / tile);
    rewrite each (v bcol) as (tid % tile);
    rewrite each v1 as (acc2 (ematrix_subtile (eA_p) tile tile (SZ.v mrow) (SZ.v !bk)) (tid / tile) (tid % tile));
    rewrite each v2 as (acc2 (ematrix_subtile (eB_p) tile tile (SZ.v !bk) (SZ.v mcol)) (tid / tile) (tid % tile));

    rewrite tensor_pts_to_cell sa1 (idx2 (tid/tile) (tid%tile)) (mapA (acc2 (ematrix_subtile (eA_p) tile tile (SZ.v mrow) (SZ.v !bk)) (tid/tile) (tid%tile))) **
            tensor_pts_to_cell sa2 (idx2 (tid/tile) (tid%tile)) (mapB (acc2 (ematrix_subtile (eB_p) tile tile (SZ.v !bk) (SZ.v mcol)) (tid/tile) (tid%tile)))
         as barrier_p_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk + 1) tid;

    assert pure (barrier_p_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk + 1) tid
                 == (shmem_contract mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2).rin (2 * !bk + 1) tid);
    rewrite barrier_p_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk + 1) tid
        as (shmem_contract mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2).rin (2 * !bk + 1) tid;

    B.barrier_wait ();

    even_2x (!bk + 1);
    assert pure (2 * (!bk + 1) == 2 * !bk + 2);
    assert pure (odd (2 * !bk + 1));
    assert pure (even (2 * !bk + 2));
    assert pure (SZ.v !bk < mshared);
    assert pure ((2 * SZ.v !bk + 1) < 2 * mshared);
    assert pure ((2 * SZ.v !bk + 1) / 2 == SZ.v !bk);

    rewrite (shmem_contract mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2).rout (2 * !bk + 1) tid
         as barrier_q_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk + 1) tid;
    rewrite barrier_q_cell mapA mapB tile (eA_p) (eB_p) (SZ.v rest) sa1 sa2 (2 * !bk + 1) tid
         as sa1 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapA (ematrix_subtile (eA_p) tile tile (SZ.v mrow) (SZ.v !bk))) **
            sa2 |-> Frac (1.0R /. (tile * tile)) (Chest.chest_map mapB (ematrix_subtile (eB_p) tile tile (SZ.v !bk) (SZ.v mcol)));

    rewrite each (tid / tile) as v brow;
    rewrite each (tid % tile) as v bcol;

    let t = Kuiper.DotProd.matmul_dotprod sa1 sa2 brow bcol;
    let s = !sum;
    sum := s `add` t;

    let sub_rA = ematrix_subtile (rA_p) tile tile (SZ.v mrow) (SZ.v !bk);
    let sub_rB = ematrix_subtile (rB_p) tile tile (SZ.v !bk) (SZ.v mcol);

    (* Mapping the subtiles into [tacc] preserves the approximation. *)
    MU.chest_map_approx mapA mapA_r
      (ematrix_subtile (eA_p) tile tile (SZ.v mrow) (SZ.v !bk)) sub_rA;
    MU.chest_map_approx mapB mapB_r
      (ematrix_subtile (eB_p) tile tile (SZ.v !bk) (SZ.v mcol)) sub_rB;

    (* t approximates the real matmul over the mapped subtiles. *)
    MU.__matmul_single_approx_real
      (Chest.chest_map mapA (ematrix_subtile (eA_p) tile tile (SZ.v mrow) (SZ.v !bk)))
      (Chest.chest_map mapB (ematrix_subtile (eB_p) tile tile (SZ.v !bk) (SZ.v mcol)))
      (Chest.chest_map mapA_r sub_rA) (Chest.chest_map mapB_r sub_rB)
      brow bcol tile;

    let r_partial = MS.__gmatmul_single 0.0R ( *. ) ( +. ) (Chest.chest_map mapA_r (rA_p)) (Chest.chest_map mapB_r (rB_p)) grow gcol (SZ.v !bk * SZ.v tile);
    let r_subtile = MS.__gmatmul_single 0.0R ( *. ) ( +. ) (Chest.chest_map mapA_r sub_rA) (Chest.chest_map mapB_r sub_rB) brow bcol tile;

    MU.__gmatmul_single_split
      (Chest.chest_map mapA_r (rA_p)) (Chest.chest_map mapB_r (rB_p))
      grow gcol (SZ.v !bk * SZ.v tile) tile
      (Chest.chest_map mapA_r sub_rA) (Chest.chest_map mapB_r sub_rB)
      brow bcol;
    assert (pure (
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) (Chest.chest_map mapA_r (rA_p)) (Chest.chest_map mapB_r (rB_p)) grow gcol (SZ.v !bk * SZ.v tile + SZ.v tile)
      == r_partial +. r_subtile));
    assert (pure ((SZ.v !bk + 1) * SZ.v tile == SZ.v !bk * SZ.v tile + SZ.v tile));

    assert (pure (2 * (!bk + 1) == 2 * !bk + 1 + 1));

    bk := !bk +^ 1sz;
    ()
  };

  (* Restore A and B page slices. *)
  elim_trade
    (gA_p |-> Frac (fA /. ((batch * (mrows * mcols)) * (tile * tile))) (eA_p))
    (gA |-> Frac (fA /. ((batch * (mrows * mcols)) * (tile * tile))) eA);
  elim_trade
    (gB_p |-> Frac (fB /. ((batch * (mrows * mcols)) * (tile * tile))) (eB_p))
    (gB |-> Frac (fB /. ((batch * (mrows * mcols)) * (tile * tile))) eB);

  (* Read/update the output cell via a concrete rank-3 index [ci] whose [up] is
     the abstract arithmetic cell [cidx] used by bkpre1/bkpost1.  The barrier
     decode rewrote the cell index into [page]/[rest] form, so the forward
     rewrite matches that expanded shape and the backward rewrite restores the
     [sbtile_cell_idx] shape that bkpost1 expects. *)
  let bidn : natlt (batch * (mrows * mcols)) = SZ.v bid;
  let tidn : natlt (tile * tile) = SZ.v tid;
  let grow_sz : szlt (mrows * tile) = mrow *^ tile +^ brow;
  let gcol_sz : szlt (mcols * tile) = mcol *^ tile +^ bcol;
  let ci : conc (batch @| (mrows * tile) @| (mcols * tile) @| INil)
         = (page, (grow_sz, (gcol_sz, ())));
  assert (pure (SZ.v grow_sz == SZ.v mrow * SZ.v tile + SZ.v brow));
  assert (pure (SZ.v gcol_sz == SZ.v mcol * SZ.v tile + SZ.v bcol));
  up3_lemma #batch #(mrows * tile) #(mcols * tile) page grow_sz gcol_sz;
  assert (pure (up ci == sbtile_cell_idx batch mrows mcols tile bidn tidn));
  rewrite (tensor_pts_to_cell gC
             ((SZ.v page <: natlt batch),
               (((SZ.v rest / SZ.v mcols * SZ.v tile + SZ.v brow) <: natlt (mrows * tile)),
                 (((SZ.v rest % SZ.v mcols * SZ.v tile + SZ.v bcol) <: natlt (mcols * tile)), ())))
             (Chest.acc eC
               ((SZ.v page <: natlt batch),
                 (((SZ.v rest / SZ.v mcols * SZ.v tile + SZ.v brow) <: natlt (mrows * tile)),
                   (((SZ.v rest % SZ.v mcols * SZ.v tile + SZ.v bcol) <: natlt (mcols * tile)), ())))))
       as (tensor_pts_to_cell gC (up ci) (Chest.acc eC (up ci)));
  let v0 = tensor_read_cell gC ci;
  let v1 = comb v0 !sum;
  tensor_write_cell gC ci v1;
  rewrite (tensor_pts_to_cell gC (up ci) v1)
       as (tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid)) v1);

  (* Bridge the cell value to the per-page real spec. *)
  acc_bridge batch mrows mcols tile rC bidn tidn;
  acc_bridge batch mrows mcols tile eC bidn tidn;
  assert (pure (Chest.acc eC (sbtile_cell_idx batch mrows mcols tile bidn tidn)
                  %~ Chest.acc rC (sbtile_cell_idx batch mrows mcols tile bidn tidn)));

  (* Restore the barrier token to the arithmetic (undecoded) form. *)
  rewrite each eA_p as (chest_slice 0 (SZ.v bid % batch) eA <: chest2 ta (mrows * tile) (mshared * tile));
  rewrite each eB_p as (chest_slice 0 (SZ.v bid % batch) eB <: chest2 tb (mshared * tile) (mcols * tile));
  rewrite each (SZ.v rest) as (SZ.v bid / batch);

  tensor_concr sa1; rewrite each core sa1 as ar1;
  tensor_concr sa2; rewrite each core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);

  fold (bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);

  (* Fold live_c_shmem for each shmem array *)
  rewrite (exists* v. pts_to (fst sh) #(1.0R /. (tile * tile)) v)
      as  (live_c_shmem (fst sh) #(1.0R /. (tile * tile)));
  rewrite (exists* v. pts_to (fst (snd sh)) #(1.0R /. (tile * tile)) v)
      as  (live_c_shmem (fst (snd sh)) #(1.0R /. (tile * tile)));

  fold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));
  fold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
  fold_live_c_shmems_cons sh #(1.0R /. (tile * tile));

  ()
}
#pop-options

(* ─── batched setup / teardown (ForEvery distribution) ────────────────────── *)

#push-options "--z3rlimit 100"
ghost
fn bsetup
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (#fA #fB : perm)
  (#eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (#eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (#eC : chest3 tc batch (mrows * tile) (mcols * tile))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt (batch *^ (mrows *^ mcols)))
             (tid : natlt (tile *^ tile)).
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp
{
  ();
  let n_threads : nat = (batch * (mrows * mcols)) * (tile * tile);

  tensor_share_n gA n_threads;
  tensor_share_n gB n_threads;
  tensor_explode gC;

  forevery_iso (sbtile_idx_bij batch mrows mcols tile)
    (fun (idx : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)) ->
       tensor_pts_to_cell gC idx (Chest.acc eC idx));
  forevery_unflatten' _;

  sbtile_gg_all batch mrows mcols tile;

  forevery_ext_2 _
   (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile bid tid)
         (Chest.acc eC (sbtile_cell_idx batch mrows mcols tile bid tid)));

  forevery_factor n_threads (batch * (mrows * mcols)) (tile * tile)
    (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (batch * (mrows * mcols)) (tile * tile)
    (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  forevery_zip3_2
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
       gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
       gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
       tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile bid tid)
         (Chest.acc eC (sbtile_cell_idx batch mrows mcols tile bid tid)));

  forevery_ext_2 _
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
       bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid);
  forevery_rw_size2
    (batch * (mrows * mcols)) (SZ.v (batch *^ (mrows *^ mcols)))
    (tile * tile) (SZ.v (tile *^ tile));
  ();
}
#pop-options

#push-options "--z3rlimit 150 --ifuel 5"
ghost
fn bteardown
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (#fA #fB : perm)
  (#eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (#eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (#eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt (batch *^ (mrows *^ mcols)))
             (tid : natlt (tile *^ tile)).
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : chest3 tc batch (mrows * tile) (mcols * tile)).
      gC |-> eC' **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  forevery_rw_size2
    (SZ.v (batch *^ (mrows *^ mcols))) (batch * (mrows * mcols))
    (SZ.v (tile *^ tile)) (tile * tile);

  forevery_unzip_2
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
       gA |-> Frac (fA /. ((batch * (mrows * mcols)) * (tile * tile))) eA) _;
  forevery_unzip_2
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
       gB |-> Frac (fB /. ((batch * (mrows * mcols)) * (tile * tile))) eB) _;

  forevery_unfactor' ((batch * (mrows * mcols)) * (tile * tile)) (batch * (mrows * mcols)) (tile * tile)
    (fun (_ : natlt (batch * (mrows * mcols))) (_ : natlt (tile * tile)) ->
       gA |-> Frac (fA /. ((batch * (mrows * mcols)) * (tile * tile))) eA);
  forevery_unfactor' ((batch * (mrows * mcols)) * (tile * tile)) (batch * (mrows * mcols)) (tile * tile)
    (fun (_ : natlt (batch * (mrows * mcols))) (_ : natlt (tile * tile)) ->
       gB |-> Frac (fB /. ((batch * (mrows * mcols)) * (tile * tile))) eB);
  tensor_gather_n gA ((batch * (mrows * mcols)) * (tile * tile));
  tensor_gather_n gB ((batch * (mrows * mcols)) * (tile * tile));

  let vf : (natlt (batch * (mrows * mcols)) -> natlt (tile * tile) -> GTot tc) =
    forevery_exists_2
      (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) (v : tc) ->
        tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile bid tid) v **
        pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
                (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
                (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
                (((bid / batch) / mcols) * tile + (tid / tile))
                (((bid / batch) % mcols) * tile + (tid % tile))));

  let eC' : chest3 tc batch (mrows * tile) (mcols * tile) =
    Chest.mk (batch @| (mrows * tile) @| (mcols * tile) @| INil)
      (fun (idx : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)) ->
        let (bid, tid) = (sbtile_idx_bij batch mrows mcols tile).ff idx in
        vf bid tid);

  forevery_extract_pure_2
    #(natlt (batch * (mrows * mcols))) #(natlt (tile * tile))
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile bid tid) (vf bid tid) **
      pure (vf bid tid %~ MS.ggemm_single mapA_r mapB_r comb_r
              (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
              (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
              (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
              (((bid / batch) / mcols) * tile + (tid / tile))
              (((bid / batch) % mcols) * tile + (tid % tile))))
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
      vf bid tid %~ MS.ggemm_single mapA_r mapB_r comb_r
        (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
        (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
        (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
        (((bid / batch) / mcols) * tile + (tid / tile))
        (((bid / batch) % mcols) * tile + (tid % tile)))
    fn bid tid { (); };

  forevery_map_2
    #(natlt (batch * (mrows * mcols))) #(natlt (tile * tile))
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile bid tid) (vf bid tid) **
      pure (vf bid tid %~ MS.ggemm_single mapA_r mapB_r comb_r
              (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
              (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
              (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
              (((bid / batch) / mcols) * tile + (tid / tile))
              (((bid / batch) % mcols) * tile + (tid % tile))))
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (sbtile_cell_idx batch mrows mcols tile bid tid) (vf bid tid))
    fn bid tid { () };

  sbtile_gg_all batch mrows mcols tile;

  forevery_ext_2 _
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((sbtile_idx_bij batch mrows mcols tile).gg (bid, tid))
        (Chest.acc eC' ((sbtile_idx_bij batch mrows mcols tile).gg (bid, tid))));
  forevery_flatten'
    (fun (xy : natlt (batch * (mrows * mcols)) & natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((sbtile_idx_bij batch mrows mcols tile).gg xy)
        (Chest.acc eC' ((sbtile_idx_bij batch mrows mcols tile).gg xy)));
  forevery_iso_back (sbtile_idx_bij batch mrows mcols tile)
    (fun (idx : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)) ->
      tensor_pts_to_cell gC idx (Chest.acc eC' idx));
  tensor_implode gC;

  (* Final batched matrix-level approximation, reduced cellwise/pagewise. *)
  bmmcomb_all_shmem mapA_r mapB_r comb_r rA rB rC;
  assert pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB);
  ();
}
#pop-options

(* ─── batched block setup / teardown (shmem fraction zip/unzip) ─────────────── *)
ghost
fn bblock_setup
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : SZ.t)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (#fA #fB : perm)
  (#eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (#eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (#eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc tacc tile))
  (bid : natlt (batch * (mrows * mcols)))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt2 tile  tile).
      bkpre mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  gpu_live_c_shmems_share_underspec sh #1.0R #(tile * tile);
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));
  forevery_zip
    (fun (tid : natlt2 tile tile) ->
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;
  ();
}

#push-options "--z3rlimit 20"
ghost
fn bblock_teardown
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : SZ.t)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA)
  (gB : array3 tb lB)
  (gC : array3 tc lC)
  (#fA #fB : perm)
  (#eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (#eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (#eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (sh : c_shmems (shmems_desc tacc tile))
  (bid : natlt (batch * (mrows * mcols)))
  ()
  norewrite
  requires
    (forall+ (tid : natlt2 tile  tile).
      bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
{
  forevery_rw_size (SZ.v (tile *^ tile)) (tile * tile);
  forevery_unzip
    (fun (tid : natlt (tile * tile)) ->
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
    _;
  gpu_live_c_shmems_gather_underspec sh #1.0R #(tile * tile);
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));
}
#pop-options

(* ─── batched sendables ────────────────────────────────────────────────────── *)
#push-options "--z3rlimit_factor 10 --fuel 1 --ifuel 1 --split_queries no"
#push-options "--z3rlimit 100"
let bkpre_block_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc tacc tile))
  (_:squash (c_shmems_inv sh))
  (bid : natlt (batch * (mrows * mcols)))
  (tid : natlt (tile * tile))
: is_send_across block_of (bkpre mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid)
= solve

let bkpost_block_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc tacc tile))
  (_:squash (c_shmems_inv sh))
  (bid : natlt (batch * (mrows * mcols)))
  (tid : natlt (tile * tile))
: is_send_across block_of (bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid)
= solve

let bblock_pre_gpu_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (batch * (mrows * mcols)))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid)
= solve

let bblock_post_gpu_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (batch * (mrows * mcols)))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
= solve
#pop-options
#pop-options

(* ─── batched kernel descriptor (the ONLY kernel description) ───────────────── *)
inline_for_extraction noextract
let bmk_kernel
  (tile : valid_tile)
  (slA slB : full_layout2 tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #mrows #mshared #mcols : szp)
  (#lA : layout3 batch (mrows   * tile) (mshared * tile))
  (#lB : layout3 batch (mshared * tile) (mcols   * tile))
  (#lC : layout3 batch (mrows   * tile) (mcols   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (#fA #fB : perm)
  (#eA : chest3 ta batch (mrows * tile) (mshared * tile))
  (#eB : chest3 tb batch (mshared * tile) (mcols * tile))
  (#eC : chest3 tc batch (mrows * tile) (mcols * tile))
  (rA : chest3 real batch (mrows   * tile) (mshared * tile))
  (rB : chest3 real batch (mshared * tile) (mcols   * tile))
  (rC : chest3 real batch (mrows   * tile) (mcols   * tile))
  (#_ : squash (SZ.fits (slA.ulen)))
  (#_ : squash (SZ.fits (slB.ulen)))
  (_ : squash (batch * (mrows * mcols) <= max_blocks
               /\ tile * tile <= max_threads
               /\ eA %~ rA /\ eB %~ rB /\ eC %~ rC
               /\ MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : chest3 tc batch (mrows * tile) (mcols * tile)).
          gC |-> eC' **
          pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB)))
= {
  nblk = batch *^ (mrows *^ mcols);
  nthr = tile *^ tile;

  barrier_contract = (fun _bid ptrs ->
    shmem_contract mapA mapB tile
      (chest_slice 0 (_bid % batch) eA <: chest2 ta (mrows * tile) (mshared * tile))
      (chest_slice 0 (_bid % batch) eB <: chest2 tb (mshared * tile) (mcols * tile))
      (_bid / batch)
      (from_array slA (fst ptrs)) (from_array slB (fst (snd ptrs))));
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun _bid ptrs ->
    barrier_p_to_q_cell_transform mapA mapB tile
      (chest_slice 0 (_bid % batch) eA <: chest2 ta (mrows * tile) (mshared * tile))
      (chest_slice 0 (_bid % batch) eB <: chest2 tb (mshared * tile) (mcols * tile))
      (_bid / batch)
      (from_array slA (fst ptrs)) (from_array slB (fst (snd ptrs))));

  shmems_desc = shmems_desc tacc tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). bkpre1  mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup      = bsetup    tile mapA mapB comb mapA_r mapB_r comb_r gA gB gC;
  teardown   = bteardown tile mapA mapB comb mapA_r mapB_r comb_r gA gB gC rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = bblock_setup    tile slA slB mapA mapB comb mapA_r mapB_r comb_r gA gB gC #_ #_ #_ #_ #eC;
  block_teardown = bblock_teardown tile slA slB mapA mapB comb mapA_r mapB_r comb_r gA gB gC #_ #_ #_ #_ #eC rA rB rC;

  kpre      = bkpre  mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost     = bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;

  f = bkf tile slA slB mapA mapB comb mapA_r mapB_r comb_r gA gB gC rA rB rC;

  block_pre_sendable=bblock_pre_gpu_sendable mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB;
  block_post_sendable=bblock_post_gpu_sendable mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;
  kpre_sendable=bkpre_block_sendable mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost_sendable=bkpost_block_sendable mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;
}

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn gbmmcomb_gpu_approx
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch (m * tile) (k * tile))
  (#lB : layout3 batch (k * tile) (n * tile))
  (#lC : layout3 batch (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (rA  : chest3 real batch (m * tile) (k * tile))
  (rB  : chest3 real batch (k * tile) (n * tile))
  (rC  : chest3 real batch (m * tile) (n * tile))
  (#eA : chest3 ta batch (m * tile) (k * tile))
  (#eB : chest3 tb batch (k * tile) (n * tile))
  (#eC : chest3 tc batch (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (bsize_req batch m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 tc batch (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  open Kuiper.Tensor.Layout.Alg;
  dassert (tile >^ 0sz);
  launch_sync (bmk_kernel tile (l2_row_major _ _) (l2_row_major _ _) mapA mapB comb mapA_r mapB_r comb_r gA gB gC rA rB rC ());
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn bmmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch (m * tile) (k * tile))
  (#lB : layout3 batch (k * tile) (n * tile))
  (#lC : layout3 batch (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array3 et lA { is_global gA })
  (gB : array3 et lB { is_global gB })
  (gC : array3 et lC { is_global gC })
  (rA  : chest3 real batch (m * tile) (k * tile))
  (rB  : chest3 real batch (k * tile) (n * tile))
  (rC  : chest3 real batch (m * tile) (n * tile))
  (#eA : chest3 et batch (m * tile) (k * tile))
  (#eB : chest3 et batch (k * tile) (n * tile))
  (#eC : chest3 et batch (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (bsize_req batch m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 et batch (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.bmmcomb comb_r rC rA rB))
{
  MS.gbmmcomb_id comb_r rC rA rB;
  gbmmcomb_gpu_approx tile (fun (x:et) -> x) (fun (x:et) -> x) comb
    (fun (r:real) -> r) (fun (r:real) -> r) comb_r gA gB gC rA rB rC;
  ()
}
#pop-options

(* [bsize_req] at batch one follows from the rank-2 size requirement. *)
let size_req_bsize1 (m n k tile : nat)
  : Lemma (requires m * n <= max_blocks) (ensures bsize_req 1 m n k tile)
  = ()

(* ─── rank-2 entry point (batch-one specialization of gbmmcomb) ───────────── *)
#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn gmmcomb_gpu_approx
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  (#lC : layout2 (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 ta lA { is_global gA })
  (gB : array2 tb lB { is_global gB })
  (gC : array2 tc lC { is_global gC })
  (rA  : chest2 real (m * tile) (k * tile))
  (rB  : chest2 real (k * tile) (n * tile))
  (rC  : chest2 real (m * tile) (n * tile))
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  (#eC : chest2 tc (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 tc (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  let afA : squash (all_fit ((m * tile) @| (k * tile) @| INil)) = C.layout2_all_fit lA;
  let afB : squash (all_fit ((k * tile) @| (n * tile) @| INil)) = C.layout2_all_fit lB;
  let afC : squash (all_fit ((m * tile) @| (n * tile) @| INil)) = C.layout2_all_fit lC;
  size_req_bsize1 (SZ.v m) (SZ.v n) (SZ.v k) (SZ.v tile);

  (* cast_in: relayout the rank-2 ownership to its batch-one rank-3 view. *)
  map_loc gpu_loc (fun () -> C.t2_to_t3n (m * tile) (k * tile) afA gA);
  map_loc gpu_loc (fun () -> C.t2_to_t3n (k * tile) (n * tile) afB gB);
  map_loc gpu_loc (fun () -> C.t2_to_t3n (m * tile) (n * tile) afC gC);

  (* carry the approximation facts to the rank-3 chests. *)
  MU.c2_to_c3_approx (m * tile) (k * tile) afA eA rA;
  MU.c2_to_c3_approx (k * tile) (n * tile) afB eB rB;
  MU.c2_to_c3_approx (m * tile) (n * tile) afC eC rC;

  gbmmcomb_gpu_approx tile mapA mapB comb mapA_r mapB_r comb_r
    #1sz #m #n #k
    #(C.l2_to_l3n #(m * tile) #(k * tile) #lA)
    #(C.l2_to_l3n #(k * tile) #(n * tile) #lB)
    #(C.l2_to_l3n #(m * tile) #(n * tile) #lC)
    (relay gA (C.l2_to_l3n #(m * tile) #(k * tile) #lA))
    (relay gB (C.l2_to_l3n #(k * tile) #(n * tile) #lB))
    (relay gC (C.l2_to_l3n #(m * tile) #(n * tile) #lC))
    (C.c2_to_c3n (m * tile) (k * tile) afA rA)
    (C.c2_to_c3n (k * tile) (n * tile) afB rB)
    (C.c2_to_c3n (m * tile) (n * tile) afC rC)
    #(C.c2_to_c3n (m * tile) (k * tile) afA eA)
    #(C.c2_to_c3n (k * tile) (n * tile) afB eB)
    #(C.c2_to_c3n (m * tile) (n * tile) afC eC)
    #fA #fB;

  (* restore the flat rank-2 views of A and B. *)
  map_loc gpu_loc (fun () -> C.t3_to_t2n_ow (m * tile) (k * tile) afA gA);
  map_loc gpu_loc (fun () -> C.t3_to_t2n_ow (k * tile) (n * tile) afB gB);

  (* cast_out for C: lower the batched result to the rank-2 gmmcomb post. *)
  with eC3'. assert (on gpu_loc (relay gC (C.l2_to_l3n #(m * tile) #(n * tile) #lC) |-> eC3'));
  map_loc gpu_loc (fun () -> C.t3_to_t2n (m * tile) (n * tile) afC gC);
  MU.c3_to_c2_approx (m * tile) (n * tile) afC eC3'
    (MS.gbmmcomb mapA_r mapB_r comb_r
      (C.c2_to_c3n (m * tile) (n * tile) afC rC)
      (C.c2_to_c3n (m * tile) (k * tile) afA rA)
      (C.c2_to_c3n (k * tile) (n * tile) afB rB));
  MU.batch1_gmmcomb mapA_r mapB_r comb_r
    (m * tile) (k * tile) (n * tile) afC afA afB rC rA rB;
  ();
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
  MS.gmmcomb_id comb_r rC rA rB;
  gmmcomb_gpu_approx tile (fun x -> x) (fun x -> x) comb (fun r -> r) (fun r -> r) comb_r gA gB gC rA rB rC;
}
#pop-options
