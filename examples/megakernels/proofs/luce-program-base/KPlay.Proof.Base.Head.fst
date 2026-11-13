module KPlay.Proof.Base.Head

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery
open FStar.Ghost

module A = Pulse.Lib.Array
module S = FStar.Seq
module C = Kuiper.Chest
module T = Kuiper.Tensor
module Warp = Kuiper.Barrier.Warp
module Every = Kuiper.ForEvery
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module MMA = Kuiper.TensorCore.MMA
module Native = KPlay.Proof.Base.NativeProjection
module Scratch = KPlay.Proof.Base.Scratch
module Transfer = KPlay.Proof.Base.Transfer
module Order = KPlay.Proof.Base.Order
module Selection = KPlay.Proof.Base.NativeSelection
module Vocabulary = KPlay.Proof.Base.Vocabulary

inline_for_extraction noextract
let lane = i:nat{i < 32}

inline_for_extraction noextract
let lane_only (i:lane) (p:slprop) : slprop =
  if i = 0 then p else emp

inline_for_extraction noextract
let selection_at (best:larray FP.t 1) (next:larray I64.t 1)
  (value:Order.selection) (i:lane) : slprop =
  lane_only i (Selection.owns best next value)

ghost
fn give_lane_zero (p:slprop) ()
  requires p
  ensures forall+ (i:lane). lane_only i p
{
  Every.forevery_intro_if #lane (0 <: lane) (fun (_:lane) -> p);
  Every.forevery_ext #lane
    (fun (i:lane) -> if Every.t2b (Prims.eq2 #lane i (0 <: lane)) then p else emp)
    (fun (i:lane) -> lane_only i p);
}

ghost
fn take_lane_zero (p:slprop) ()
  requires forall+ (i:lane). lane_only i p
  ensures p
{
  Every.forevery_extract_if_eqtype #lane (0 <: lane) (fun (i:lane) -> lane_only i p);
  Every.forevery_ext #lane
    (fun (i:lane) -> if ( = ) #lane i (0 <: lane) then emp else lane_only i p)
    (fun (_:lane) -> emp);
  Every.forevery_emp_elim lane;
  unfold lane_only 0 p;
}

ghost
fn gather_hidden (hidden:array2 BF.t Native.hidden_layout)
  (#values:C.chest2 BF.t 16000 1) ()
  requires forall+ (i:lane). hidden |-> Frac (1.0R /. 32) values
  ensures forall+ (i:lane). lane_only i (hidden |-> values)
{
  T.tensor_gather_n hidden 32;
  give_lane_zero (hidden |-> values) ();
}

ghost
fn scatter_hidden (hidden:array2 BF.t Native.hidden_layout)
  (#values:C.chest2 BF.t 16000 1) ()
  requires forall+ (i:lane). lane_only i (hidden |-> values)
  ensures forall+ (i:lane). hidden |-> Frac (1.0R /. 32) values
{
  take_lane_zero (hidden |-> values) ();
  T.tensor_share_n hidden 32;
}

inline_for_extraction noextract
let gather_wait (hidden:array2 BF.t Native.hidden_layout)
  (#values:C.chest2 BF.t 16000 1) (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** hidden |-> Frac (1.0R /. 32) values)
      (ensures fun _ -> thread_id 32 tid ** lane_only tid (hidden |-> values)) =
  Warp.warp_barrier_wait ()
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) values)
    (fun (i:lane) -> lane_only i (hidden |-> values))
    (gather_hidden hidden #values ()) #32 #tid

inline_for_extraction noextract
let scatter_wait (hidden:array2 BF.t Native.hidden_layout)
  (#values:C.chest2 BF.t 16000 1) (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** lane_only tid (hidden |-> values))
      (ensures fun _ -> thread_id 32 tid ** hidden |-> Frac (1.0R /. 32) values) =
  Warp.warp_barrier_wait ()
    (fun (i:lane) -> lane_only i (hidden |-> values))
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) values)
    (scatter_hidden hidden #values ()) #32 #tid

inline_for_extraction noextract
let flat_hidden (value:C.chest2 BF.t 16000 1)
  : Lemma (T.to_seq Native.hidden_layout value == Native.hidden_values value) =
  C.ext (T.from_seq Native.hidden_layout (Native.hidden_values value)) value;
  T.to_from Native.hidden_layout (Native.hidden_values value)

ghost
fn hidden_to_array (hidden:array2 BF.t Native.hidden_layout)
  (#value:C.chest2 BF.t 16000 1) (#fraction:perm)
  requires hidden |-> Frac fraction value
  ensures A.pts_to (T.core hidden) #fraction (Native.hidden_values value)
{
  flat_hidden value;
  T.tensor_concr hidden;
  rewrite A.pts_to (T.core hidden) #fraction (T.to_seq Native.hidden_layout value)
    as A.pts_to (T.core hidden) #fraction (Native.hidden_values value);
}

ghost
fn array_to_hidden (hidden:array2 BF.t Native.hidden_layout)
  (#value:C.chest2 BF.t 16000 1) (#fraction:perm)
  requires A.pts_to (T.core hidden) #fraction (Native.hidden_values value)
  ensures hidden |-> Frac fraction value
{
  flat_hidden value;
  rewrite A.pts_to (T.core hidden) #fraction (Native.hidden_values value)
    as A.pts_to (T.core hidden) #fraction (T.to_seq Native.hidden_layout value);
  T.tensor_abs Native.hidden_layout (T.core hidden) #fraction #value;
  T.lem_core_from_array hidden;
  rewrite each (T.from_array Native.hidden_layout (T.core hidden)) as hidden;
}

inline_for_extraction noextract
let produced (weights:C.chest2 BF.t 752393024 1)
  (hidden:C.chest2 BF.t 16000 1) (row:Transfer.row_block) =
  Transfer.block_accumulator (Native.packed_values weights) (Native.hidden_values hidden) row 64

inline_for_extraction noextract
let published (weights:C.chest2 BF.t 752393024 1)
  (hidden:C.chest2 BF.t 16000 1) (row:Transfer.row_block) =
  Scratch.replace_window hidden 12500 16 (Native.stored_column (produced weights hidden row))

inline_for_extraction noextract
let selected (weights:C.chest2 BF.t 752393024 1)
  (hidden:C.chest2 BF.t 16000 1) (row:Transfer.row_block) (before:Order.selection) =
  Order.scan (Transfer.consumed_scores (produced weights hidden row)) row before 16

(* One actual final-projection row block, including both source __syncwarp
   publications and the lane-zero selector. The predicates do not depend on
   the calling lane; all 32 lanes use the same row and source snapshots. *)
inline_for_extraction noextract
fn final_block
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (weight:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1)
  (#before:erased Selection.bounded_selection)
  (#weight_fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weight |-> Frac weight_fraction vw
  requires hidden |-> Frac (1.0R /. 32) vh
  requires selection_at best next before (SZ.v tid)
  ensures hidden |-> Frac (1.0R /. 32) (published vw vh (SZ.v row))
  ensures selection_at best next (selected vw vh (SZ.v row) before) (SZ.v tid)
{
  let accum = MMA.alloc_fragment ();
  let part = MMA.alloc_fragment ();
  Scratch.owned_projection_block row weight hidden accum part;
  gather_wait hidden #(published vw vh (SZ.v row)) #(SZ.v tid);
  if (tid <^ 1sz) {
    assert pure (SZ.v tid == 0);
    rewrite lane_only (SZ.v tid) (hidden |-> published vw vh (SZ.v row))
      as hidden |-> published vw vh (SZ.v row);
    rewrite selection_at best next before (SZ.v tid) as Selection.owns best next before;
    hidden_to_array hidden;
    Selection.scan_source_block (T.core hidden) best next row;
    Selection.producer_consumer_binding vh (produced vw vh (SZ.v row));
    rewrite Selection.owns best next
      (Order.scan (Selection.source_values (Native.hidden_values (published vw vh (SZ.v row))))
        (SZ.v row) before 16)
      as Selection.owns best next (selected vw vh (SZ.v row) before);
    array_to_hidden hidden;
    rewrite hidden |-> published vw vh (SZ.v row)
      as lane_only (SZ.v tid) (hidden |-> published vw vh (SZ.v row));
    rewrite Selection.owns best next (selected vw vh (SZ.v row) before)
      as selection_at best next (selected vw vh (SZ.v row) before) (SZ.v tid);
  } else {
    rewrite selection_at best next before (SZ.v tid)
      as selection_at best next (selected vw vh (SZ.v row) before) (SZ.v tid);
  };
  scatter_wait hidden #(published vw vh (SZ.v row)) #(SZ.v tid);
  with av. assert accum |-> av;
  with pv. assert part |-> pv;
  drop_ (accum |-> av);
  drop_ (part |-> pv);
}

inline_for_extraction noextract
let prefix_bound (weights:Transfer.packed) (hidden:Transfer.scratch)
  (count:nat{count <= 15520})
  : Lemma ((Vocabulary.scan_blocks weights hidden count).Order.token < 248320) =
  Vocabulary.scan_blocks_agree weights hidden count;
  Order.scan_dominates (Vocabulary.score_values weights hidden) 0 Vocabulary.seed (16 * count)

inline_for_extraction noextract
let block_step (vw:C.chest2 BF.t 752393024 1) (vh:C.chest2 BF.t 16000 1)
  (count:nat{count < 15520})
  : Lemma (
      let weights = Native.packed_values vw in
      let initial = Native.hidden_values vh in
      let current = Vocabulary.scratch_after weights vh count in
      published vw current (16 * count) == Vocabulary.scratch_after weights vh (count + 1) /\
      selected vw current (16 * count) (Vocabulary.scan_blocks weights initial count) ==
        Vocabulary.scan_blocks weights initial (count + 1)) =
  let weights = Native.packed_values vw in
  let initial = Native.hidden_values vh in
  Vocabulary.scratch_step weights vh count;
  norm_spec [delta_once ["KPlay.Proof.Base.Vocabulary.scan_blocks"]]
    (Vocabulary.scan_blocks weights initial (count + 1))

inline_for_extraction noextract
fn final_projection
  (weight:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1)
  (#weight_fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weight |-> Frac weight_fraction vw
  requires hidden |-> Frac (1.0R /. 32) vh
  requires selection_at best next Vocabulary.seed (SZ.v tid)
  ensures hidden |-> Frac (1.0R /. 32)
    (Vocabulary.scratch_after (Native.packed_values vw) vh 15520)
  ensures selection_at best next
    (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) 15520) (SZ.v tid)
{
  let mut row = 0sz;
  while (!row <^ SZ.uint_to_t 248320)
    invariant exists* (r:sz{SZ.v r <= 248320 /\ SZ.v r % 16 == 0}).
      row |-> r **
      hidden |-> Frac (1.0R /. 32)
        (Vocabulary.scratch_after (Native.packed_values vw) vh (SZ.v r / 16)) **
      selection_at best next
        (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) (SZ.v r / 16))
        (SZ.v tid)
    invariant gpu ** thread_id 32 (SZ.v tid) ** weight |-> Frac weight_fraction vw
    decreases (248320 - SZ.v !row)
  {
    let r = !row;
    assert pure (SZ.v r == 16 * (SZ.v r / 16));
    prefix_bound (Native.packed_values vw) (Native.hidden_values vh) (SZ.v r / 16);
    final_block r weight hidden best next tid;
    block_step vw vh (SZ.v r / 16);
    let following = SZ.add r 16sz;
    assert pure (SZ.v following / 16 == SZ.v r / 16 + 1);
    rewrite hidden |-> Frac (1.0R /. 32)
      (published vw (Vocabulary.scratch_after (Native.packed_values vw) vh (SZ.v r / 16)) (SZ.v r))
      as hidden |-> Frac (1.0R /. 32)
        (Vocabulary.scratch_after (Native.packed_values vw) vh (SZ.v following / 16));
    rewrite selection_at best next
      (selected vw (Vocabulary.scratch_after (Native.packed_values vw) vh (SZ.v r / 16)) (SZ.v r)
        (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) (SZ.v r / 16)))
      (SZ.v tid)
      as selection_at best next
        (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) (SZ.v following / 16))
        (SZ.v tid);
    row := following;
  };
  let r = !row;
  assert pure (SZ.v r == 248320 /\ SZ.v r / 16 == 15520);
  rewrite selection_at best next
    (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) (SZ.v r / 16))
    (SZ.v tid)
    as selection_at best next
      (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) 15520) (SZ.v tid);
}
