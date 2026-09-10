module Kuiper.Kernel.GEMM.FlipFlopColumns

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Chest
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Math { even, odd }
module B = Kuiper.Barrier
module SZ = Kuiper.SizeT

(* A column-wise flip-flop barrier. Odd phases preserve the expected tile values. *)

let own_1_col
  (#et : Type0)
  (#tile : valid_tile)
  (#l : layout2 tile tile)
  (m : array2 et l)
  (tid : natlt tile)
  : slprop =
  forall+ (ii : natlt tile).
    exists* (x : et).
      Cell m (idx2 ii tid) |-> x

let own_1_col_at
  (#et : Type0) (#tile : valid_tile)
  (#l : layout2 tile tile)
  (m : array2 et l) (em : chest2 et tile tile) (tid : natlt tile)
  : slprop =
  forall+ (ii : natlt tile).
    tensor_pts_to_cell m (idx2 ii tid) (acc2 em ii tid)

let barrier_p
  (#et : Type0) (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  : B.barrier_side tile =
  fun it tid ->
    if even it then
      (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)
    else
      own_1_col_at m1 (v1 (it / 2)) tid **
      own_1_col_at m2 (v2 (it / 2)) tid

let barrier_q
  (#et : Type0) (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  : B.barrier_side tile =
  fun it tid ->
    if even it then own_1_col m1 tid ** own_1_col m2 tid
    else m1 |-> Frac (1.0R /. tile) (v1 (it / 2)) **
         m2 |-> Frac (1.0R /. tile) (v2 (it / 2))

let barrier_contract
  (#et : Type0)
  (tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (* This is defined over the base shared larrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  : B.contract tile =
  {
    rin  = barrier_p v1 v2 (from_array l1 ar1) (from_array l2 ar2);
    rout = barrier_q v1 v2 (from_array l1 ar1) (from_array l2 ar2);
  }

(* Projecting [rin]/[rout] out of the record literal built by [barrier_contract]
   is no longer reduced for the SMT solver, so discharge those slprop equalities
   by normalization instead. *)
let unfold_barrier_contract () : FStar.Tactics.V2.Tac unit =
  FStar.Tactics.V2.norm [delta_only [`%barrier_contract]; iota; primops];
  Pulse.Lib.Core.slprop_equiv_norm ()

(* Per-tid fold/unfold helpers that collapse the symbolic [even it] match in
   [barrier_p] via a runtime [if even it], discharging the impossible branch
   with [unreachable].  Needed because Pulse's [rewrite] cannot evaluate
   [even (2*bk)] / [odd (2*bk+1)] on a symbolic [bk]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
ghost
fn fold_barrier_p_even
  (#et : Type0)
  (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    ((exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
     (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)) **
    pure (even it)
  ensures
    barrier_p v1 v2 m1 m2 it tid
{
  let ev = even it;
  if ev {
    rewrite (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
            (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)
         as barrier_p v1 v2 m1 m2 it tid;
  } else {
    unreachable ();
  }
}

ghost
fn fold_barrier_p_odd
  (#et : Type0) (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat) (tid : natlt tile)
  requires
    own_1_col_at m1 (v1 (it / 2)) tid **
    own_1_col_at m2 (v2 (it / 2)) tid ** pure (odd it)
  ensures barrier_p v1 v2 m1 m2 it tid
{
  let ev = even it;
  if ev { unreachable (); }
  else {
    rewrite own_1_col_at m1 (v1 (it / 2)) tid **
            own_1_col_at m2 (v2 (it / 2)) tid
      as barrier_p v1 v2 m1 m2 it tid;
  }
}

ghost
fn unfold_barrier_q_even
  (#et : Type0) (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat) (tid : natlt tile)
  requires barrier_q v1 v2 m1 m2 it tid ** pure (even it)
  ensures own_1_col m1 tid ** own_1_col m2 tid
{
  let ev = even it;
  if ev {
    rewrite barrier_q v1 v2 m1 m2 it tid as own_1_col m1 tid ** own_1_col m2 tid;
  } else { unreachable (); }
}

ghost
fn unfold_barrier_q_odd
  (#et : Type0) (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat) (tid : natlt tile)
  requires barrier_q v1 v2 m1 m2 it tid ** pure (odd it)
  ensures m1 |-> Frac (1.0R /. tile) (v1 (it / 2)) **
          m2 |-> Frac (1.0R /. tile) (v2 (it / 2))
{
  let ev = even it;
  if ev { unreachable (); }
  else {
    rewrite barrier_q v1 v2 m1 m2 it tid as
      m1 |-> Frac (1.0R /. tile) (v1 (it / 2)) **
      m2 |-> Frac (1.0R /. tile) (v2 (it / 2));
  }
}

(* Bridge helpers connecting [barrier_p v1 v2 sa1 sa2] (held over the *raised*
   shared arrays [sa1 = from_array l1 ar1], [sa2 = from_array l2 ar2]) to the
   barrier contract's [.rin]/[.rout], which are stated over the *raw* arrays
   [ar1 ar2].  We use [rewrite each] (syntactic substitution, requiring only the
   pure equality) to swap [sa1]<->[from_array l1 ar1] before the otherwise
   reflexive contract rewrite. *)
ghost
fn barrier_p_to_rin
  (#et : Type0)
  (tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (sa1 : array2 et l1) (sa2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    barrier_p v1 v2 sa1 sa2 it tid **
    pure (sa1 == from_array l1 ar1 /\ sa2 == from_array l2 ar2)
  ensures
    (barrier_contract tile v1 v2 l1 l2 ar1 ar2).rin it tid
{
  rewrite each sa1 as (from_array l1 ar1);
  rewrite each sa2 as (from_array l2 ar2);
  rewrite barrier_p v1 v2 (from_array l1 ar1) (from_array l2 ar2) it tid
       as (barrier_contract tile v1 v2 l1 l2 ar1 ar2).rin it tid
       by unfold_barrier_contract ();
}

ghost
fn rout_to_barrier_q
  (#et : Type0)
  (tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (sa1 : array2 et l1) (sa2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    (barrier_contract tile v1 v2 l1 l2 ar1 ar2).rout it tid **
    pure (sa1 == from_array l1 ar1 /\ sa2 == from_array l2 ar2)
  ensures
    barrier_q v1 v2 sa1 sa2 it tid
{
  rewrite (barrier_contract tile v1 v2 l1 l2 ar1 ar2).rout it tid
       as barrier_q v1 v2 (from_array l1 ar1) (from_array l2 ar2) it tid
       by unfold_barrier_contract ();
  rewrite each (from_array l1 ar1) as sa1;
  rewrite each (from_array l2 ar2) as sa2;
}
#pop-options

(* ---- Barrier transform proof ---- *)

(* Even → odd: distribute fractional whole-array ownership into per-column cells. *)
ghost
fn even_barrier_p_to_q
  (#et : Type0)
  (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat{even it})
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  requires
    forall+ (tid : natlt tile). barrier_p v1 v2 m1 m2 it tid
  ensures
    forall+ (tid : natlt tile). barrier_q v1 v2 m1 m2 it tid
{
  assert pure (even it);
  (* barrier_p even = frac shares; barrier_q even = own_1_col *)
  forevery_map
    (fun (tid : natlt tile) -> barrier_p v1 v2 m1 m2 it tid)
    (fun (tid : natlt tile) ->
      (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x))
    fn tid {
      rewrite barrier_p v1 v2 m1 m2 it tid
           as (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
              (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x);
    };
  forevery_unzip _ _;
  tensor_gather_n_underspec m1 tile;
  tensor_gather_n_underspec m2 tile;
  with em1. assert (m1 |-> em1);
  with em2. assert (m2 |-> em2);
  tensor_ilower2 m1;
  tensor_ilower2 m2;
  forevery_commute (fun (r c : natlt tile) -> tensor_pts_to_cell m1 (idx2 r c) (acc2 em1 r c));
  forevery_commute (fun (r c : natlt tile) -> tensor_pts_to_cell m2 (idx2 r c) (acc2 em2 r c));
  forevery_map
    (fun (c : natlt tile) -> forall+ (r : natlt tile). tensor_pts_to_cell m1 (idx2 r c) (acc2 em1 r c))
    (fun (c : natlt tile) -> own_1_col m1 c)
    fn c {
      forevery_map
        (fun (r : natlt tile) -> tensor_pts_to_cell m1 (idx2 r c) (acc2 em1 r c))
        (fun (r : natlt tile) -> exists* (x : et). Cell m1 (idx2 r c) |-> x)
        fn r { };
      fold own_1_col m1 c;
    };
  forevery_map
    (fun (c : natlt tile) -> forall+ (r : natlt tile). tensor_pts_to_cell m2 (idx2 r c) (acc2 em2 r c))
    (fun (c : natlt tile) -> own_1_col m2 c)
    fn c {
      forevery_map
        (fun (r : natlt tile) -> tensor_pts_to_cell m2 (idx2 r c) (acc2 em2 r c))
        (fun (r : natlt tile) -> exists* (x : et). Cell m2 (idx2 r c) |-> x)
        fn r { };
      fold own_1_col m2 c;
    };
  forevery_zip
    (fun (tid : natlt tile) -> own_1_col m1 tid)
    (fun (tid : natlt tile) -> own_1_col m2 tid);
  forevery_map
    (fun (tid : natlt tile) -> own_1_col m1 tid ** own_1_col m2 tid)
    (fun (tid : natlt tile) -> barrier_q v1 v2 m1 m2 it tid)
    fn tid {
      rewrite own_1_col m1 tid ** own_1_col m2 tid
           as barrier_q v1 v2 m1 m2 it tid;
    };
}

(* Odd → even: collect per-column cells back to fractional whole-array ownership. *)
ghost
fn odd_barrier_p_to_q
  (#et : Type0) (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat{odd it})
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  requires forall+ (tid : natlt tile). barrier_p v1 v2 m1 m2 it tid
  ensures forall+ (tid : natlt tile). barrier_q v1 v2 m1 m2 it tid
{
  forevery_map
    (fun tid -> barrier_p v1 v2 m1 m2 it tid)
    (fun tid -> own_1_col_at m1 (v1 (it / 2)) tid ** own_1_col_at m2 (v2 (it / 2)) tid)
    fn tid {
      rewrite barrier_p v1 v2 m1 m2 it tid as
        own_1_col_at m1 (v1 (it / 2)) tid ** own_1_col_at m2 (v2 (it / 2)) tid;
    };
  forevery_unzip _ _;
  forevery_map
    (fun c -> own_1_col_at m1 (v1 (it / 2)) c)
    (fun c -> forall+ (r : natlt tile). tensor_pts_to_cell m1 (idx2 r c) (acc2 (v1 (it / 2)) r c))
    fn c { unfold own_1_col_at m1 (v1 (it / 2)) c };
  forevery_map
    (fun c -> own_1_col_at m2 (v2 (it / 2)) c)
    (fun c -> forall+ (r : natlt tile). tensor_pts_to_cell m2 (idx2 r c) (acc2 (v2 (it / 2)) r c))
    fn c { unfold own_1_col_at m2 (v2 (it / 2)) c };
  forevery_commute (fun (c r : natlt tile) -> tensor_pts_to_cell m1 (idx2 r c) (acc2 (v1 (it / 2)) r c));
  forevery_commute (fun (c r : natlt tile) -> tensor_pts_to_cell m2 (idx2 r c) (acc2 (v2 (it / 2)) r c));
  tensor_iraise2 m1;
  tensor_iraise2 m2;
  tensor_share_n m1 tile;
  tensor_share_n m2 tile;
  forevery_zip (fun (_ : natlt tile) -> m1 |-> Frac (1.0R /. tile) (v1 (it / 2))) _;
  forevery_map
    (fun (_ : natlt tile) -> m1 |-> Frac (1.0R /. tile) (v1 (it / 2)) ** m2 |-> Frac (1.0R /. tile) (v2 (it / 2)))
    (fun tid -> barrier_q v1 v2 m1 m2 it tid)
    fn tid {
      rewrite m1 |-> Frac (1.0R /. tile) (v1 (it / 2)) ** m2 |-> Frac (1.0R /. tile) (v2 (it / 2))
        as barrier_q v1 v2 m1 m2 it tid;
    };
}

(* Both helpers have the same pre/postcondition shape (barrier_p → barrier_q),
   so we can define the barrier_transform directly by case-splitting on even/odd.
   We use a regular F* let to avoid Pulse's if/else effect promotion issue. *)
#push-options "--z3rlimit 80"
let barrier_p_to_q_transform
  (#et : Type0)
  (#tile : valid_tile)
  (v1 v2 : nat -> GTot (chest2 et tile tile))
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  : B.barrier_transform (barrier_contract tile v1 v2 l1 l2 ar1 ar2)
  = let m1 = from_array l1 ar1 in
    let m2 = from_array l2 ar2 in
    fun (it : nat) ->
      if even it then
        even_barrier_p_to_q v1 v2 m1 m2 it
      else
        odd_barrier_p_to_q v1 v2 m1 m2 it
#pop-options
