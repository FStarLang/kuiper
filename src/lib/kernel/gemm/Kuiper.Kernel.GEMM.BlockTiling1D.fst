module Kuiper.Kernel.GEMM.BlockTiling1D

#lang-pulse

#set-options "--z3rlimit 40"

open Kuiper
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Slice
open Pulse.Lib.Trade
open Kuiper.Chest
open Kuiper.Bijection

module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module C = Kuiper.Matrix.Casts
module Chest = Kuiper.Chest

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |} (tile:valid_tile) : list shmem_desc = [
  SHArray et (tile *^ tile);
  SHArray et (tile *^ tile);
]

(* The barrier flip-flops between an initial state
where every threads shares all of the array, and
a second state where each thread owns two cells
of the array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)

(* To verify functional correctness: the existentials here should be made
precise, and parametrize this over the starting input matrices. *)
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

let barrier_p
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  : B.barrier_side tile =
  fun it tid ->
    if even it then
      (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)
    else
      own_1_col m1 tid ** own_1_col m2 tid

let barrier_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  : B.barrier_side tile =
  fun it tid -> barrier_p m1 m2 (it+1) tid (* flip flop *)

let barrier_contract
  (#et : Type0)
  (tile : valid_tile)
  (* This is defined over the base shared larrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  : B.contract tile =
  {
    rin  = barrier_p (from_array l1 ar1) (from_array l2 ar2);
    rout = barrier_q (from_array l1 ar1) (from_array l2 ar2);
  }

(* Per-tid fold/unfold helpers that collapse the symbolic [even it] match in
   [barrier_p] via a runtime [if even it], discharging the impossible branch
   with [unreachable].  Needed because Pulse's [rewrite] cannot evaluate
   [even (2*bk)] / [odd (2*bk+1)] on a symbolic [bk]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
ghost
fn fold_barrier_p_even
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    ((exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
     (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)) **
    pure (even it)
  ensures
    barrier_p m1 m2 it tid
{
  let ev = even it;
  if ev {
    rewrite (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
            (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)
         as barrier_p m1 m2 it tid;
  } else {
    unreachable ();
  }
}

ghost
fn unfold_barrier_p_even
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    barrier_p m1 m2 it tid ** pure (even it)
  ensures
    (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
    (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)
{
  let ev = even it;
  if ev {
    rewrite barrier_p m1 m2 it tid
         as (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
            (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x);
  } else {
    unreachable ();
  }
}

ghost
fn fold_barrier_p_odd
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    (own_1_col m1 tid ** own_1_col m2 tid) ** pure (odd it)
  ensures
    barrier_p m1 m2 it tid
{
  let ev = even it;
  if ev {
    unreachable ();
  } else {
    rewrite own_1_col m1 tid ** own_1_col m2 tid
         as barrier_p m1 m2 it tid;
  }
}

ghost
fn unfold_barrier_p_odd
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    barrier_p m1 m2 it tid ** pure (odd it)
  ensures
    own_1_col m1 tid ** own_1_col m2 tid
{
  let ev = even it;
  if ev {
    unreachable ();
  } else {
    rewrite barrier_p m1 m2 it tid
         as own_1_col m1 tid ** own_1_col m2 tid;
  }
}

(* Bridge helpers connecting [barrier_p sa1 sa2] (held over the *raised*
   shared arrays [sa1 = from_array l1 ar1], [sa2 = from_array l2 ar2]) to the
   barrier contract's [.rin]/[.rout], which are stated over the *raw* arrays
   [ar1 ar2].  We use [rewrite each] (syntactic substitution, requiring only the
   pure equality) to swap [sa1]<->[from_array l1 ar1] before the otherwise
   reflexive contract rewrite. *)
ghost
fn barrier_p_to_rin
  (#et : Type0)
  (tile : valid_tile)
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (sa1 : array2 et l1) (sa2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    barrier_p sa1 sa2 it tid **
    pure (sa1 == from_array l1 ar1 /\ sa2 == from_array l2 ar2)
  ensures
    (barrier_contract tile l1 l2 ar1 ar2).rin it tid
{
  rewrite each sa1 as (from_array l1 ar1);
  rewrite each sa2 as (from_array l2 ar2);
  rewrite barrier_p (from_array l1 ar1) (from_array l2 ar2) it tid
       as (barrier_contract tile l1 l2 ar1 ar2).rin it tid;
}

ghost
fn rout_to_barrier_p
  (#et : Type0)
  (tile : valid_tile)
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (sa1 : array2 et l1) (sa2 : array2 et l2)
  (it : nat)
  (tid : natlt tile)
  requires
    (barrier_contract tile l1 l2 ar1 ar2).rout it tid **
    pure (sa1 == from_array l1 ar1 /\ sa2 == from_array l2 ar2)
  ensures
    barrier_p sa1 sa2 (it + 1) tid
{
  rewrite (barrier_contract tile l1 l2 ar1 ar2).rout it tid
       as barrier_q (from_array l1 ar1) (from_array l2 ar2) it tid;
  rewrite each (from_array l1 ar1) as sa1;
  rewrite each (from_array l2 ar2) as sa2;
  rewrite barrier_q sa1 sa2 it tid
       as barrier_p sa1 sa2 (it + 1) tid;
}
#pop-options

(* ---- Barrier transform proof ---- *)

(* Even → odd: distribute fractional whole-array ownership into per-column cells. *)
ghost
fn even_barrier_p_to_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat{even it})
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  requires
    forall+ (tid : natlt tile). barrier_p m1 m2 it tid
  ensures
    forall+ (tid : natlt tile). barrier_q m1 m2 it tid
{
  assert pure (even it);
  (* barrier_p even = frac shares; barrier_q even = own_1_col *)
  forevery_map
    (fun (tid : natlt tile) -> barrier_p m1 m2 it tid)
    (fun (tid : natlt tile) ->
      (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x))
    fn tid {
      rewrite barrier_p m1 m2 it tid
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
  forevery_commute (fun (r : natlt tile) (c : natlt tile) -> tensor_pts_to_cell m1 (idx2 r c) (acc2 em1 r c));
  forevery_commute (fun (r : natlt tile) (c : natlt tile) -> tensor_pts_to_cell m2 (idx2 r c) (acc2 em2 r c));
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
    (fun (tid : natlt tile) -> barrier_q m1 m2 it tid)
    fn tid {
      rewrite own_1_col m1 tid ** own_1_col m2 tid
           as barrier_q m1 m2 it tid;
    };
}

(* Odd → even: collect per-column cells back to fractional whole-array ownership. *)
ghost
fn odd_barrier_p_to_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : layout2 tile tile) (m1 : array2 et l1)
  (#l2 : layout2 tile tile) (m2 : array2 et l2)
  (it : nat{odd it})
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  requires
    forall+ (tid : natlt tile). barrier_p m1 m2 it tid
  ensures
    forall+ (tid : natlt tile). barrier_q m1 m2 it tid
{
  assert pure (odd it);
  (* barrier_p odd = own_1_col; barrier_q odd = frac shares *)
  forevery_map
    (fun (tid : natlt tile) -> barrier_p m1 m2 it tid)
    (fun (tid : natlt tile) -> own_1_col m1 tid ** own_1_col m2 tid)
    fn tid {
      rewrite barrier_p m1 m2 it tid
           as own_1_col m1 tid ** own_1_col m2 tid;
    };
  forevery_unzip _ _;
  (* Unfold own_1_col to nested forall+/exists *)
  forevery_map
    (fun (c : natlt tile) -> own_1_col m1 c)
    (fun (c : natlt tile) -> forall+ (r : natlt tile). exists* (x : et). Cell m1 (idx2 r c) |-> x)
    fn c { unfold own_1_col m1 c };
  forevery_map
    (fun (c : natlt tile) -> own_1_col m2 c)
    (fun (c : natlt tile) -> forall+ (r : natlt tile). exists* (x : et). Cell m2 (idx2 r c) |-> x)
    fn c { unfold own_1_col m2 c };
  (* Commute: forall+ c r -> forall+ r c *)
  forevery_commute (fun (c : natlt tile) (r : natlt tile) -> exists* (x : et). Cell m1 (idx2 r c) |-> x);
  forevery_commute (fun (c : natlt tile) (r : natlt tile) -> exists* (x : et). Cell m2 (idx2 r c) |-> x);
  (* Extract witnesses *)
  let f1 = forevery_exists_2 (fun (r : natlt tile) (c : natlt tile) (x : et) -> Cell m1 (idx2 r c) |-> x);
  let f2 = forevery_exists_2 (fun (r : natlt tile) (c : natlt tile) (x : et) -> Cell m2 (idx2 r c) |-> x);
  (* Construct ematrices from witness functions *)
  let em1 : chest2 et tile tile = mk2 f1;
  let em2 : chest2 et tile tile = mk2 f2;
  (* Rewrite cells to use acc2 *)
  forevery_map
    (fun (r : natlt tile) -> forall+ (c : natlt tile). Cell m1 (idx2 r c) |-> f1 r c)
    (fun (r : natlt tile) -> forall+ (c : natlt tile). tensor_pts_to_cell m1 (idx2 r c) (acc2 em1 r c))
    fn r {
      forevery_ext
        (fun (c : natlt tile) -> Cell m1 (idx2 r c) |-> f1 r c)
        (fun (c : natlt tile) -> tensor_pts_to_cell m1 (idx2 r c) (acc2 em1 r c));
    };
  forevery_map
    (fun (r : natlt tile) -> forall+ (c : natlt tile). Cell m2 (idx2 r c) |-> f2 r c)
    (fun (r : natlt tile) -> forall+ (c : natlt tile). tensor_pts_to_cell m2 (idx2 r c) (acc2 em2 r c))
    fn r {
      forevery_ext
        (fun (c : natlt tile) -> Cell m2 (idx2 r c) |-> f2 r c)
        (fun (c : natlt tile) -> tensor_pts_to_cell m2 (idx2 r c) (acc2 em2 r c));
    };
  tensor_iraise2 m1;
  tensor_iraise2 m2;
  tensor_share_n m1 tile;
  tensor_share_n m2 tile;
  forevery_zip
    (fun (_ : natlt tile) -> m1 |-> Frac (1.0R /. tile) em1) _;
  forevery_map
    (fun (tid : natlt tile) ->
      m1 |-> Frac (1.0R /. tile) em1 **
      m2 |-> Frac (1.0R /. tile) em2)
    (fun (tid : natlt tile) ->
      (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x))
    fn tid { };
  forevery_map
    (fun (tid : natlt tile) ->
      (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x))
    (fun (tid : natlt tile) -> barrier_q m1 m2 it tid)
    fn tid {
      rewrite
        (exists* (x : chest2 _ _ _). m1 |-> Frac (1.0R /. tile) x) **
        (exists* (x : chest2 _ _ _). m2 |-> Frac (1.0R /. tile) x)
      as
        barrier_q m1 m2 it tid;
    };
}

(* Both helpers have the same pre/postcondition shape (barrier_p → barrier_q),
   so we can define the barrier_transform directly by case-splitting on even/odd.
   We use a regular F* let to avoid Pulse's if/else effect promotion issue. *)
#push-options "--z3rlimit 80"
let barrier_p_to_q_transform
  (#et : Type0)
  (#tile : valid_tile)
  (l1 l2 : full_layout2 tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  : B.barrier_transform (barrier_contract tile l1 l2 ar1 ar2)
  = let m1 = from_array l1 ar1 in
    let m2 = from_array l2 ar2 in
    fun (it : nat) ->
      if even it then
        even_barrier_p_to_q m1 m2 it
      else
        odd_barrier_p_to_q m1 m2 it
#pop-options

inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#ta #tb #tacc : Type0) {| scalar ta, scalar tb, scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (#m #n #k : erased nat)
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : array2 ta lA)
  (gB : array2 tb lB)
  (#l1 #l2 : layout2 tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (sa1 : array2 tacc l1) (sa2 : array2 tacc l2)
  (mm : szlt m)
  (kk : szlt k)
  (nn : szlt n)
  (tid : szlt tile)
  (#fA #fB : perm)
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  (* Should have stronger spec. *)
  requires
    own_1_col sa1 tid **
    own_1_col sa2 tid
  ensures
    own_1_col sa1 tid **
    own_1_col sa2 tid
{
  let mut i = 0sz;
  while (!i <^ tile)
    invariant live i
    decreases (tile - !i)
  {
    {
      unfold own_1_col sa1 tid;
      forevery_extract #(natlt tile) !i _;
      let tileA = array2_extract_tile_ro' gA (SZ.v tile) (SZ.v tile) (SZ.v mm) (SZ.v kk);
      let ci = !i;
      let v1 = tensor_read tileA ((ci <: szlt _), ((tid <: szlt _), ()));
      tensor_write_cell sa1 ((ci <: szlt _), ((tid <: szlt _), ())) (mapA v1);
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold own_1_col sa1 tid;
    };

    {
      unfold own_1_col sa2 tid;
      forevery_extract #(natlt tile) !i _;
      let tileB = array2_extract_tile_ro' gB (SZ.v tile) (SZ.v tile) (SZ.v kk) (SZ.v nn);
      let ci = !i;
      let v2 = tensor_read tileB ((ci <: szlt _), ((tid <: szlt _), ()));
      tensor_write_cell sa2 ((ci <: szlt _), ((tid <: szlt _), ())) (mapB v2);
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold own_1_col sa2 tid;
    };

    i := !i +^ 1sz;
  }
}

inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : layout2 tile tile) (#l2 : layout2 tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (m1 : array2 et l1)
  (m2 : array2 et l2)
  (j : szlt tile)
  (#acc0 : erased (lseq et tile))
  (#v1 #v2 : chest2 et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    acc |-> acc0
  ensures
    exists* (acc' : lseq et tile).
      acc |-> acc'
{
  pts_to_len acc;
  let mut sk : sz = 0sz;
  while (!sk <^ tile)
    invariant live sk ** live acc
    decreases (tile - !sk)
  {
    let mut i = 0sz;
    (* We can read v2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let csk = !sk;
    let v2 = tensor_read m2 ((csk <: szlt _), ((j <: szlt _), ()));
    while (!i <^ tile)
      invariant live i ** live acc
      decreases (tile - !i)
    {
      let ci = !i;
      let csk2 = !sk;
      let v1 = tensor_read m1 ((ci <: szlt _), ((csk2 <: szlt _), ()));

      open Pulse.Lib.Array;
      pts_to_len acc;
      acc.(!i) <- acc.(!i) `add` (v1 `mul` v2);
      i := !i +^ 1sz;
    };
    sk := !sk +^ 1sz;
  };
  pts_to_len acc;
}

(* ═══════════════════════════════════════════════════════════════════════════
   BATCHED (rank-3) KERNEL

   The batched kernel is the ONLY real kernel description; the rank-2 entry below
   is derived from it at [batch = 1].  Each block fixes a page (batch index),
   slices the rank-3 operands down to their rank-2 page views, and reuses the
   exact same content-agnostic barrier + shared-memory protocol as the rank-2
   body.  Threads remain columns ([nthr = tile]); the block grid is PAGE-MINOR:
   bid = rest * batch + page, rest = mrow * mcols + mcol.
   ═══════════════════════════════════════════════════════════════════════════ *)

(* ─── batched index bijection (rank-3 output, tiled, column-per-thread) ──────
   Maps the abstract rank-3 output index [(page,(grow,(gcol,())))] to the nested
   pair [(bid, (tid, ii))], with block id PAGE-MINOR and each thread [tid]
   owning a column [ii] of the output tile:
     page = bid % batch,
     grow = ((bid / batch) / mcols) * tile + ii,
     gcol = ((bid / batch) % mcols) * tile + tid. *)

let sbij_comm_size (a b : nat) : (natlt (a * b) =~ natlt (b * a)) =
  {
    ff = (fun (x : natlt (a * b)) -> (x <: natlt (b * a)));
    gg = (fun (x : natlt (b * a)) -> (x <: natlt (a * b)));
    ff_gg = (fun x -> ());
    gg_ff = (fun x -> ());
  }

let sabs_shuffle (batch gr gc : nat)
  : (abs (batch @| gr @| gc @| INil) =~ ((natlt gr & natlt gc) & natlt batch))
  = {
      ff = (fun (pg, (r, (c, ()))) -> ((r, c), pg));
      gg = (fun ((r, c), pg) -> (pg, (r, (c, ()))));
      ff_gg = (fun ((r, c), pg) -> ());
      gg_ff = (fun (pg, (r, (c, ()))) -> ());
    }

let sabs_regroup (mrows mcols tile batch : nat)
  : (((natlt mrows & natlt tile) & (natlt mcols & natlt tile)) & natlt batch
     =~ ((natlt mrows & natlt mcols) & natlt batch) & (natlt tile & natlt tile))
  = {
      ff = (fun (((mr, br), (mc, bc)), pg) -> (((mr, mc), pg), (br, bc)));
      gg = (fun (((mr, mc), pg), (br, bc)) -> (((mr, br), (mc, bc)), pg));
      ff_gg = (fun (((mr, mc), pg), (br, bc)) -> ());
      gg_ff = (fun (((mr, br), (mc, bc)), pg) -> ());
    }

let sflat_block (mrows mcols batch : nat)
  : ((natlt mrows & natlt mcols) & natlt batch =~ natlt (batch * (mrows * mcols)))
  = bij_comp
      (bij_prod (bij_nat_prod #mrows #mcols) (bij_self (natlt batch)))
      (bij_comp (bij_nat_prod #(mrows * mcols) #batch)
                (sbij_comm_size (mrows * mcols) batch))

(* Final factor keeps [(tid, ii)] as a pair (columns), flipping the natural
   [(brow, bcol)] = [(ii, tid)] order into [(tid, ii)]. *)
let bbtile_idx_bij (batch mrows mcols tile : nat)
  : (abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)
     =~ natlt (batch * (mrows * mcols)) & (natlt tile & natlt tile))
  = bij_comp (sabs_shuffle batch (mrows * tile) (mcols * tile))
      (bij_comp
         (bij_prod
            (bij_prod (bij_sym (bij_nat_prod #mrows #tile))
                      (bij_sym (bij_nat_prod #mcols #tile)))
            (bij_self (natlt batch)))
         (bij_comp (sabs_regroup mrows mcols tile batch)
            (bij_prod (sflat_block mrows mcols batch)
                      (bij_flip #(natlt tile) #(natlt tile)))))

(* The direct page-minor arithmetic cell index for the output tensor.
   [prod_ff] supplies the [< mrows*tile] / [< mcols*tile] bounds for free. *)
unfold
let bbtile_cell_idx (batch mrows mcols tile : nat)
  (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) (ii : natlt tile)
  : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)
  = let page = bid % batch in
    let rest = bid / batch in
    let mrow = rest / mcols in
    let mcol = rest % mcols in
    ((page <: natlt batch),
      ((prod_ff mrows tile ((mrow <: natlt mrows), (ii <: natlt tile))),
        ((prod_ff mcols tile ((mcol <: natlt mcols), (tid <: natlt tile))), ())))

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
  (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) (ii : natlt tile)
  : Lemma (
      Chest.acc e (bbtile_cell_idx batch mrows mcols tile bid tid ii)
        == acc2 (slice_page e (bid % batch))
             (((bid / batch) / mcols) * tile + ii)
             (((bid / batch) % mcols) * tile + tid))
  = ()
#pop-options

(* [up] of an explicit rank-3 concrete cell index reduces componentwise. *)
#push-options "--fuel 4 --ifuel 4"
let up3_lemma (#b #r #cc : nat) (p : szlt b) (g : szlt r) (co : szlt cc)
  : Lemma (up ((p, (g, (co, ()))) <: conc (b @| r @| cc @| INil))
             == ((SZ.v p <: natlt b), ((SZ.v g <: natlt r), ((SZ.v co <: natlt cc), ()))))
  = ()
#pop-options

(* The full block/thread/col bijection decodes to the direct arithmetic cell. *)
let bbtile_gg_full (batch mrows mcols tile : nat)
  (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) (ii : natlt tile)
  : Lemma ((bbtile_idx_bij batch mrows mcols tile).gg (bid, (tid, ii))
             == bbtile_cell_idx batch mrows mcols tile bid tid ii)
  = assert_norm ((bbtile_idx_bij batch mrows mcols tile).gg (bid, (tid, ii))
                   == bbtile_cell_idx batch mrows mcols tile bid tid ii)

let bbtile_gg_all (batch mrows mcols tile : nat)
  : Lemma (forall (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) (ii : natlt tile).
             (bbtile_idx_bij batch mrows mcols tile).gg (bid, (tid, ii))
               == bbtile_cell_idx batch mrows mcols tile bid tid ii)
  = introduce
      forall (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) (ii : natlt tile).
        (bbtile_idx_bij batch mrows mcols tile).gg (bid, (tid, ii))
          == bbtile_cell_idx batch mrows mcols tile bid tid ii
      with bbtile_gg_full batch mrows mcols tile bid tid ii

(* Threadcell form of the bijection: collapse the [(tid,ii)] pair into a single
   [natlt (tile*tile)], so that [bteardown] can reuse the verified 2-level
   [forevery_exists_2]/[Chest.mk] machinery (mirroring SHMem). *)
let bbtile_idx_bij2 (batch mrows mcols tile : nat)
  : (abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)
     =~ natlt (batch * (mrows * mcols)) & natlt (tile * tile))
  = bij_comp (bbtile_idx_bij batch mrows mcols tile)
      (bij_prod (bij_self (natlt (batch * (mrows * mcols)))) (bij_nat_prod #tile #tile))

let bbtile_gg2_full (batch mrows mcols tile : nat)
  (bid : natlt (batch * (mrows * mcols))) (tc : natlt (tile * tile))
  : Lemma ((bbtile_idx_bij2 batch mrows mcols tile).gg (bid, tc)
             == bbtile_cell_idx batch mrows mcols tile bid (tc / tile) (tc % tile))
  = assert_norm ((bbtile_idx_bij2 batch mrows mcols tile).gg (bid, tc)
                   == (bbtile_idx_bij batch mrows mcols tile).gg (bid, (tc / tile, tc % tile)));
    bbtile_gg_full batch mrows mcols tile bid (tc / tile) (tc % tile)

let bbtile_gg2_all (batch mrows mcols tile : nat)
  : Lemma (forall (bid : natlt (batch * (mrows * mcols))) (tc : natlt (tile * tile)).
             (bbtile_idx_bij2 batch mrows mcols tile).gg (bid, tc)
               == bbtile_cell_idx batch mrows mcols tile bid (tc / tile) (tc % tile))
  = introduce
      forall (bid : natlt (batch * (mrows * mcols))) (tc : natlt (tile * tile)).
        (bbtile_idx_bij2 batch mrows mcols tile).gg (bid, tc)
          == bbtile_cell_idx batch mrows mcols tile bid (tc / tile) (tc % tile)
      with bbtile_gg2_full batch mrows mcols tile bid tc

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

(* ─── batched per-block predicates ─────────────────────────────────────────── *)
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
  (tid : natlt tile)
  : slprop
  =
  (gA |-> Frac (fA /. ((batch * (mrows * mcols)) * tile)) eA) **
  (gB |-> Frac (fB /. ((batch * (mrows * mcols)) * tile)) eB) **
  forall+ (ii : natlt tile).
    (exists* v.
      tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v)

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
  (tid : natlt tile)
  : slprop
  =
  let page = bid % batch in
  let rest = bid / batch in
  let mrow = rest / mcols in
  let mcol = rest % mcols in
  let rA_p : chest2 real (mrows * tile) (mshared * tile) = chest_slice 0 page rA in
  let rB_p : chest2 real (mshared * tile) (mcols * tile) = chest_slice 0 page rB in
  let rC_p : chest2 real (mrows * tile) (mcols * tile) = chest_slice 0 page rC in
  (gA |-> Frac (fA /. ((batch * (mrows * mcols)) * tile)) eA) **
  (gB |-> Frac (fB /. ((batch * (mrows * mcols)) * tile)) eB) **
  forall+ (ii : natlt tile).
    exists* (v : tc).
      tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v **
      pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r rA_p rB_p rC_p
              (mrow * tile + ii) (mcol * tile + tid))

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
  (tid : natlt tile)
  : slprop
  =
  bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid **
  live_c_shmems sh #(1.0R /. tile)

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
  (tid : natlt tile)
  : slprop
  =
  bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
  live_c_shmems sh #(1.0R /. tile)

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
  (#_sq : squash (eA %~ rA /\ eB %~ rB))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc tacc tile))
  (bid : szlt (batch * (mrows * mcols)))
  (tid : szlt tile)
  ()
  norewrite
  requires
    gpu **
    bkpre mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id tile tid **
    block_id (batch * (mrows * mcols)) bid **
    B.barrier_tok (barrier_contract tile slA slB (fst sh) (fst (snd sh))) **
    B.barrier_state 0
  ensures
    gpu **
    bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid **
    thread_id tile tid **
    block_id (batch * (mrows * mcols)) bid **
    B.barrier_tok (barrier_contract tile slA slB (fst sh) (fst (snd sh))) **
    B.barrier_state (2 * mshared)
{
  unfold_c_shmems sh #(1.0R /. Real.of_int (v tile)) (`%shmems_desc);
  let (ar1, (ar2, _)) = sh;

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  tensor_abs' slA ar1;
  let sa1 = from_array slA ar1;
  rewrite each from_array slA ar1 as sa1;

  tensor_abs' slB ar2;
  let sa2 = from_array slB ar2;
  rewrite each from_array slB ar2 as sa2;

  (* Decode the page-minor block indices. *)
  let page : szlt batch = bid %^ batch;
  let rest = bid /^ batch;
  let mrow, mcol = s_divmod mcols rest;
  let bcol = tid;
  assert rewrites_to bcol tid;
  assert (pure (SZ.v page == SZ.v bid % batch));
  assert (pure (SZ.v rest == SZ.v bid / batch));
  assert (pure (SZ.v mrow == (SZ.v bid / batch) / mcols));
  assert (pure (SZ.v mcol == (SZ.v bid / batch) % mcols));

  (* Ascribed page-slice views of the real operands. *)
  let rA_p : chest2 real (mrows * tile) (mshared * tile) = chest_slice 0 (SZ.v page) rA;
  let rB_p : chest2 real (mshared * tile) (mcols * tile) = chest_slice 0 (SZ.v page) rB;
  let rC_p : chest2 real (mrows * tile) (mcols * tile) = chest_slice 0 (SZ.v page) rC;

  (* Slice out the [page]-th rank-2 page views of A and B (read-only). *)
  tensor_extract_slice_ro gA 0 (SZ.v page);
  tensor_extract_slice_ro gB 0 (SZ.v page);
  let gA_p = sliceof gA 0 (SZ.v page);
  rewrite each sliceof gA 0 (SZ.v page) as gA_p;
  let gB_p = sliceof gB 0 (SZ.v page);
  rewrite each sliceof gB 0 (SZ.v page) as gB_p;

  (* thread-local result cache *)
  let mut sums : Pulse.Lib.Array.array tacc = [| zero ; tile |];
  let mut bk = 0sz;

  while (!bk <^ mshared)
    invariant live sums
    invariant live bk ** pure (!bk <= mshared) ** B.barrier_state (2 * !bk)
    invariant
        (exists* (x : chest2 _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
        (exists* (x : chest2 _ _ _). sa2 |-> Frac (1.0R /. tile) x)
    decreases (mshared - !bk)
  {
    pts_to_len sums;

    assert pure (sa1 == from_array slA ar1 /\ sa2 == from_array slB ar2);

    // Even step: fold frac-shares -> barrier_p (2bk), hand to barrier, take own back.
    even_2x !bk;
    fold_barrier_p_even sa1 sa2 (2 * !bk) tid;
    barrier_p_to_rin tile slA slB ar1 ar2 sa1 sa2 (2 * !bk) tid;
    B.barrier_wait ();
    rout_to_barrier_p tile slA slB ar1 ar2 sa1 sa2 (2 * !bk) tid;
    odd_2x1 !bk;
    unfold_barrier_p_odd sa1 sa2 (2 * !bk + 1) tid;

    (* We exclusively own a full column of the SHMEM cache. Populate it,
       applying the input pre-maps [mapA]/[mapB] on store. *)
    bring_2cols tile mapA mapB gA_p gB_p sa1 sa2 mrow !bk mcol tid;

    odd_2x1 !bk;
    fold_barrier_p_odd sa1 sa2 (2 * !bk + 1) tid;
    barrier_p_to_rin tile slA slB ar1 ar2 sa1 sa2 (2 * !bk + 1) tid;
    B.barrier_wait ();
    rout_to_barrier_p tile slA slB ar1 ar2 sa1 sa2 (2 * !bk + 1) tid;
    rewrite barrier_p sa1 sa2 (2 * !bk + 1 + 1) tid
         as barrier_p sa1 sa2 (2 * !bk + 2) tid;
    even_2x (SZ.v !bk + 1);
    unfold_barrier_p_even sa1 sa2 (2 * !bk + 2) tid;

    (* The SHMem cache is filled with the submatrices and we have RO
       permission to it. Compute product for our column and add to sum. *)
    subproduct_cols tile sums sa1 sa2 bcol;

    (* Move to next tile *)
    bk := !bk +^ 1sz;
  };

  (* Restore A and B page slices. *)
  elim_trade
    (gA_p |-> Frac (fA /. ((batch * (mrows * mcols)) * tile)) (chest_slice 0 (SZ.v page) eA))
    (gA |-> Frac (fA /. ((batch * (mrows * mcols)) * tile)) eA);
  elim_trade
    (gB_p |-> Frac (fB /. ((batch * (mrows * mcols)) * tile)) (chest_slice 0 (SZ.v page) eB))
    (gB |-> Frac (fB /. ((batch * (mrows * mcols)) * tile)) eB);

  (* Write all the accumulated sums into the rank-3 output cells owned
     directly (by [bbtile_cell_idx]) in bkpre1. *)
  let mut row : sz = 0sz;
  pts_to_len sums;
  while (!row <^ tile)
    invariant live row ** live sums
    decreases (tile - !row)
  {
    pts_to_len sums;
    forevery_extract #(natlt tile) (!row) _;

    let crow = !row;
    let grow_sz : szlt (mrows * tile) = mrow *^ tile +^ crow;
    let gcol_sz : szlt (mcols * tile) = mcol *^ tile +^ tid;
    let ci : conc (batch @| (mrows * tile) @| (mcols * tile) @| INil) = (page, (grow_sz, (gcol_sz, ())));
    assert rewrites_to ci (page, (grow_sz, (gcol_sz, ())));
    up3_lemma #batch #(mrows * tile) #(mcols * tile) page grow_sz gcol_sz;
    assert (pure (up ci == bbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid) (SZ.v crow)));

    with v0'.
      rewrite (tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid) (SZ.v crow)) v0')
           as (tensor_pts_to_cell gC (up ci) v0');

    let v0 = tensor_read_cell gC (page, (grow_sz, (gcol_sz, ())));
    open Pulse.Lib.Array;
    let v1 = sums.(!row);
    let v' = comb v0 v1;
    tensor_write_cell gC (page, (grow_sz, (gcol_sz, ()))) v';

    rewrite (tensor_pts_to_cell gC (up ci) v')
         as (tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid) (SZ.v crow)) v');

    row := !row +^ 1sz;
    Pulse.Lib.Trade.elim_trade _ _;
  };

  tensor_concr sa1; rewrite each core sa1 as ar1;
  tensor_concr sa2; rewrite each core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  fold_c_shmems sh #(1.0R /. Real.of_int (v tile)) (`%shmems_desc);

  (* Functional correctness assumption (cf. SHMem.fst line 363, page-sliced form).
     The accumulated subproduct_cols results, combined with the old cell values,
     approximate the general (fused-map, multi-type) real gemm spec over the
     [page]-th slice.  This is the ONLY assume in the file; all else is proved. *)
  forevery_map
    (fun (ii : natlt tile) ->
      exists* (v : tc).
        tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid) ii) v)
    (fun (ii : natlt tile) ->
      exists* (v : tc).
        tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid) ii) v **
        pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r rA_p rB_p rC_p
                (SZ.v mrow * tile + ii) (SZ.v mcol * tile + tid)))
    fn ii {
      with v. assert (tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile (SZ.v bid) (SZ.v tid) ii) v);
      assume pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r rA_p rB_p rC_p
                     (SZ.v mrow * tile + ii) (SZ.v mcol * tile + tid));
    };

  (* Bridge the let-bound page-slice names / decoded indices into the expanded
     arithmetic form that [bkpost1] unfolds to (provable equalities asserted above). *)
  rewrite each rA_p as (chest_slice 0 (SZ.v page) rA <: chest2 real (mrows * tile) (mshared * tile));
  rewrite each rB_p as (chest_slice 0 (SZ.v page) rB <: chest2 real (mshared * tile) (mcols * tile));
  rewrite each rC_p as (chest_slice 0 (SZ.v page) rC <: chest2 real (mrows * tile) (mcols * tile));
  rewrite each (SZ.v mrow) as ((SZ.v bid / SZ.v batch) / SZ.v mcols);
  rewrite each (SZ.v mcol) as ((SZ.v bid / SZ.v batch) % SZ.v mcols);
  rewrite each (SZ.v page) as (SZ.v bid % SZ.v batch);
}
#pop-options

(* ─── batched setup / teardown (ForEvery distribution) ────────────────────── *)

#push-options "--z3rlimit 120 --fuel 2 --ifuel 2"
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
             (tid : natlt tile).
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp
{
  ();
  let n_threads : nat = (batch * (mrows * mcols)) * tile;

  tensor_share_n gA n_threads;
  tensor_share_n gB n_threads;
  tensor_explode gC;

  forevery_iso (bbtile_idx_bij batch mrows mcols tile)
    (fun (idx : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)) ->
       tensor_pts_to_cell gC idx (Chest.acc eC idx));
  forevery_unflatten' _;

  bbtile_gg_all batch mrows mcols tile;

  (* Per-block: split the [(tid,ii)] pair, rewrite the cell index to the direct
     arithmetic form, and weaken the concrete value to an existential. *)
  forevery_map
    (fun (bid : natlt (batch * (mrows * mcols))) ->
       forall+ (tt : natlt tile & natlt tile).
         tensor_pts_to_cell gC ((bbtile_idx_bij batch mrows mcols tile).gg (bid, tt))
           (Chest.acc eC ((bbtile_idx_bij batch mrows mcols tile).gg (bid, tt))))
    (fun (bid : natlt (batch * (mrows * mcols))) ->
       forall+ (tid : natlt tile) (ii : natlt tile).
         exists* (v : tc). tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v)
    fn bid {
      forevery_unflatten' _;
      forevery_ext_2 _
        (fun (tid : natlt tile) (ii : natlt tile) ->
           tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii)
             (Chest.acc eC (bbtile_cell_idx batch mrows mcols tile bid tid ii)));
      forevery_map_2
        (fun (tid : natlt tile) (ii : natlt tile) ->
           tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii)
             (Chest.acc eC (bbtile_cell_idx batch mrows mcols tile bid tid ii)))
        (fun (tid : natlt tile) (ii : natlt tile) ->
           exists* (v : tc). tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v)
        fn tid ii { (); };
    };

  forevery_factor n_threads (batch * (mrows * mcols)) tile
    (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (batch * (mrows * mcols)) tile
    (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  forevery_zip3_2
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) ->
       gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) ->
       gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) ->
       forall+ (ii : natlt tile).
         exists* (v : tc). tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v);

  forevery_ext_2 _
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) ->
       bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid);
  forevery_rw_size2
    (batch * (mrows * mcols)) (SZ.v (batch *^ (mrows *^ mcols)))
    tile tile;
  ();
}
#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 5"
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
             (tid : natlt tile).
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : chest3 tc batch (mrows * tile) (mcols * tile)).
      gC |-> eC' **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  let n_threads : nat = (batch * (mrows * mcols)) * tile;

  forevery_rw_size2
    (SZ.v (batch *^ (mrows *^ mcols))) (batch * (mrows * mcols))
    tile tile;

  (* Unfold bkpost1 to explicit form. *)
  forevery_ext_2
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) ->
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
    (fun (bid : natlt (batch * (mrows * mcols))) (tid : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (ii : natlt tile).
        exists* (v : tc).
          tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v **
          pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                  (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
                  (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
                  (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
                  (((bid / batch) / mcols) * tile + ii)
                  (((bid / batch) % mcols) * tile + tid)));

  (* Peel off the A/B shares. *)
  forevery_unzip_2
    (fun (_ : natlt (batch * (mrows * mcols))) (_ : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA) _;
  forevery_unzip_2
    (fun (_ : natlt (batch * (mrows * mcols))) (_ : natlt tile) ->
      gB |-> Frac (fB /. n_threads) eB) _;

  forevery_unfactor' n_threads (batch * (mrows * mcols)) tile
    (fun (_ : natlt (batch * (mrows * mcols))) (_ : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA);
  tensor_gather_n gA n_threads;
  forevery_unfactor' n_threads (batch * (mrows * mcols)) tile
    (fun (_ : natlt (batch * (mrows * mcols))) (_ : natlt tile) ->
      gB |-> Frac (fB /. n_threads) eB);
  tensor_gather_n gB n_threads;

  (* Collapse the inner (tid,ii) column into a single threadcell index, so the
     verified 2-level [forevery_exists_2]/[Chest.mk] sequence applies. *)
  forevery_map
    (fun (bid : natlt (batch * (mrows * mcols))) ->
      forall+ (tid : natlt tile) (ii : natlt tile).
        exists* (v : tc).
          tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v **
          pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                  (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
                  (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
                  (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
                  (((bid / batch) / mcols) * tile + ii)
                  (((bid / batch) % mcols) * tile + tid)))
    (fun (bid : natlt (batch * (mrows * mcols))) ->
      forall+ (tcell : natlt (tile * tile)).
        exists* (v : tc).
          tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid (tcell / tile) (tcell % tile)) v **
          pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                  (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
                  (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
                  (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
                  (((bid / batch) / mcols) * tile + (tcell % tile))
                  (((bid / batch) % mcols) * tile + (tcell / tile))))
    fn bid {
      forevery_unfactor' (tile * tile) tile tile
        (fun (tid : natlt tile) (ii : natlt tile) ->
          exists* (v : tc).
            tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid tid ii) v **
            pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                    (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
                    (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
                    (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
                    (((bid / batch) / mcols) * tile + ii)
                    (((bid / batch) % mcols) * tile + tid)));
    };

  let vf : (natlt (batch * (mrows * mcols)) -> natlt (tile * tile) -> GTot tc) =
    forevery_exists_2
      (fun (bid : natlt (batch * (mrows * mcols))) (tcell : natlt (tile * tile)) (v : tc) ->
        tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid (tcell / tile) (tcell % tile)) v **
        pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
                (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
                (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
                (((bid / batch) / mcols) * tile + (tcell % tile))
                (((bid / batch) % mcols) * tile + (tcell / tile))));

  bbtile_gg2_all batch mrows mcols tile;

  let eC' : chest3 tc batch (mrows * tile) (mcols * tile) =
    Chest.mk (batch @| (mrows * tile) @| (mcols * tile) @| INil)
      (fun (idx : abs (batch @| (mrows * tile) @| (mcols * tile) @| INil)) ->
        let (bid, tcell) = (bbtile_idx_bij2 batch mrows mcols tile).ff idx in
        vf bid tcell);

  forevery_extract_pure_2
    #(natlt (batch * (mrows * mcols))) #(natlt (tile * tile))
    (fun (bid : natlt (batch * (mrows * mcols))) (tcell : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid (tcell / tile) (tcell % tile)) (vf bid tcell) **
      pure (vf bid tcell %~ MS.ggemm_single mapA_r mapB_r comb_r
              (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
              (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
              (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
              (((bid / batch) / mcols) * tile + (tcell % tile))
              (((bid / batch) % mcols) * tile + (tcell / tile))))
    (fun (bid : natlt (batch * (mrows * mcols))) (tcell : natlt (tile * tile)) ->
      vf bid tcell %~ MS.ggemm_single mapA_r mapB_r comb_r
        (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
        (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
        (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
        (((bid / batch) / mcols) * tile + (tcell % tile))
        (((bid / batch) % mcols) * tile + (tcell / tile)))
    fn bid tcell { (); };

  forevery_map_2
    #(natlt (batch * (mrows * mcols))) #(natlt (tile * tile))
    (fun (bid : natlt (batch * (mrows * mcols))) (tcell : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid (tcell / tile) (tcell % tile)) (vf bid tcell) **
      pure (vf bid tcell %~ MS.ggemm_single mapA_r mapB_r comb_r
              (chest_slice 0 (bid % batch) rA <: chest2 real (mrows * tile) (mshared * tile))
              (chest_slice 0 (bid % batch) rB <: chest2 real (mshared * tile) (mcols * tile))
              (chest_slice 0 (bid % batch) rC <: chest2 real (mrows * tile) (mcols * tile))
              (((bid / batch) / mcols) * tile + (tcell % tile))
              (((bid / batch) % mcols) * tile + (tcell / tile))))
    (fun (bid : natlt (batch * (mrows * mcols))) (tcell : natlt (tile * tile)) ->
      tensor_pts_to_cell gC (bbtile_cell_idx batch mrows mcols tile bid (tcell / tile) (tcell % tile)) (vf bid tcell))
    fn bid tcell { () };

  forevery_ext_2 _
    (fun (bid : natlt (batch * (mrows * mcols))) (tcell : natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((bbtile_idx_bij2 batch mrows mcols tile).gg (bid, tcell))
        (Chest.acc eC' ((bbtile_idx_bij2 batch mrows mcols tile).gg (bid, tcell))));
  forevery_flatten'
    (fun (xy : natlt (batch * (mrows * mcols)) & natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((bbtile_idx_bij2 batch mrows mcols tile).gg xy)
        (Chest.acc eC' ((bbtile_idx_bij2 batch mrows mcols tile).gg xy)));
  forevery_iso_back (bbtile_idx_bij2 batch mrows mcols tile)
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
    (forall+ (tid : natlt tile).
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt tile).
      bkpre mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  gpu_live_c_shmems_share_underspec sh #1.0R #tile;
  forevery_zip
    (fun (tid : natlt tile) ->
      bkpre1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;
  ();
}

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
    (forall+ (tid : natlt tile).
      bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
{
  forevery_unzip
    (fun (tid : natlt tile) ->
      bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
    _;
  gpu_live_c_shmems_gather_underspec sh #1.0R #tile;
}

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
  (i : natlt (batch * (mrows * mcols)))
  (j : natlt tile)
: is_send_across block_of (bkpre mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB sh i j)
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
  (i : natlt (batch * (mrows * mcols)))
  (j : natlt tile)
: is_send_across block_of (bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB sh i j)
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
    (forall+ (tid : natlt tile).
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
    (forall+ (tid : natlt tile).
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
  (_ : squash (batch * (mrows * mcols) <= max_blocks
               /\ tile <= max_threads
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
  nthr = tile;

  (* Barrier fields are CONTENT-AGNOSTIC (independent of the page), so they
     coincide with the rank-2 kernel's barrier fields. *)
  barrier_contract = (fun _bid ptrs -> barrier_contract tile slA slB (fst ptrs) (fst (snd ptrs)));
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun _bid ptrs -> barrier_p_to_q_transform slA slB (fst ptrs) (fst (snd ptrs)));

  shmems_desc = shmems_desc tacc tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt tile). bkpre1  mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt tile). bkpost1 mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup      = bsetup    tile mapA mapB comb mapA_r mapB_r comb_r gA gB gC;
  teardown   = bteardown tile mapA mapB comb mapA_r mapB_r comb_r gA gB gC rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = bblock_setup    tile slA slB mapA mapB comb mapA_r mapB_r comb_r gA gB gC #_ #_ #_ #_ #eC;
  block_teardown = bblock_teardown tile slA slB mapA mapB comb mapA_r mapB_r comb_r gA gB gC #_ #_ #_ #_ #eC rA rB rC;

  kpre      = bkpre  mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost     = bkpost mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;

  f = bkf tile slA slB mapA mapB comb mapA_r mapB_r comb_r gA gB gC rA rB rC;

  block_pre_sendable=bblock_pre_gpu_sendable mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC fA fB;
  block_post_sendable=bblock_post_gpu_sendable mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpre_sendable=bkpre_block_sendable mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost_sendable=bkpost_block_sendable mapA mapB comb mapA_r mapB_r comb_r tile slA slB gA gB gC eA eB eC rA rB rC fA fB;
}

(* ─── general (fused-map, multi-type) natively batched rank-3 entry ─────────── *)
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

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
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
