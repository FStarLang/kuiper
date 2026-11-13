module KPlay.Proof.Base.WarpHidden

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery

module A = Pulse.Lib.Array
module C = Kuiper.Chest
module T = Kuiper.Tensor
module Every = Kuiper.ForEvery
module Warp = Kuiper.Barrier.Warp
module BF = Kuiper.BFloat16
module Native = KPlay.Proof.Base.NativeProjection
module Head = KPlay.Proof.Base.Head

let lane = Head.lane
let snapshot = C.chest2 BF.t 16000 1

inline_for_extraction noextract
let full (hidden:array2 BF.t Native.hidden_layout) (property:snapshot -> prop) : slprop =
  exists* (values:snapshot). hidden |-> values ** pure (property values)

inline_for_extraction noextract
let shared (hidden:array2 BF.t Native.hidden_layout) (property:snapshot -> prop) (i:lane) : slprop =
  exists* (values:snapshot). hidden |-> Frac (1.0R /. 32) values **
    pure (i = 0 ==> property values)

inline_for_extraction noextract
let nonzero (i:lane) (p:slprop) : slprop =
  if ( = ) #lane i (0 <: lane) then emp else p

inline_for_extraction noextract
let anchor (hidden:array2 BF.t Native.hidden_layout) (values:snapshot) : slprop =
  hidden |-> Frac (1.0R /. 32) values

inline_for_extraction noextract
let values_agree (left right:snapshot)
  : Lemma
      (requires Native.hidden_values left == Native.hidden_values right)
      (ensures left == right) =
  let at_index (i:nat{i < 16000}) (j:nat{j < 1})
    : Lemma (C.acc2 left i j == C.acc2 right i j) =
    assert (FStar.Seq.index (Native.hidden_values left) i == C.acc2 left i 0);
    assert (FStar.Seq.index (Native.hidden_values right) i == C.acc2 right i 0)
  in
  FStar.Classical.forall_intro_2 at_index;
  C.ext left right

ghost
fn agreement
  (hidden:array2 BF.t Native.hidden_layout) (#left #right:snapshot) (#fl #fr:perm)
  preserves hidden |-> Frac fl left ** hidden |-> Frac fr right
  ensures pure (left == right)
{
  Head.hidden_to_array hidden #left #fl;
  Head.hidden_to_array hidden #right #fr;
  A.pts_to_injective_eq #BF.t #fl #fr #(Native.hidden_values left) #(Native.hidden_values right) (T.core hidden);
  values_agree left right;
  Head.array_to_hidden hidden #left #fl;
  Head.array_to_hidden hidden #right #fr;
}

ghost
fn gather
  (hidden:array2 BF.t Native.hidden_layout) (property:snapshot -> prop) ()
  requires forall+ (i:lane). shared hidden property i
  ensures forall+ (i:lane). Head.lane_only i (full hidden property)
{
  Every.forevery_extract_if_eqtype #lane (0 <: lane) (shared hidden property);
  unfold shared hidden property (0 <: lane);
  with first. assert hidden |-> Frac (1.0R /. 32) first;
  fold anchor hidden first;
  Every.forevery_ext #lane
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else shared hidden property i)
    (fun i -> nonzero i (shared hidden property i));
  Every.forevery_map_extra #lane (anchor hidden first)
    (fun i -> nonzero i (shared hidden property i))
    (fun i -> nonzero i (hidden |-> Frac (1.0R /. 32) first))
    fn i {
      if (( = ) #lane i (0 <: lane)) {
        rewrite nonzero i (shared hidden property i)
          as nonzero i (hidden |-> Frac (1.0R /. 32) first);
      } else {
        rewrite nonzero i (shared hidden property i) as shared hidden property i;
        unfold shared hidden property i;
        with other. assert hidden |-> Frac (1.0R /. 32) other;
        unfold anchor hidden first;
        agreement hidden #first #other #(1.0R /. 32) #(1.0R /. 32);
        rewrite hidden |-> Frac (1.0R /. 32) other as hidden |-> Frac (1.0R /. 32) first;
        fold anchor hidden first;
        rewrite hidden |-> Frac (1.0R /. 32) first
          as nonzero i (hidden |-> Frac (1.0R /. 32) first);
      }
    };
  Every.forevery_ext #lane
    (fun i -> nonzero i (hidden |-> Frac (1.0R /. 32) first))
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else hidden |-> Frac (1.0R /. 32) first);
  unfold anchor hidden first;
  Every.forevery_unextract_if_eqtype #lane (0 <: lane)
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) first);
  T.tensor_gather_n hidden 32;
  fold full hidden property;
  Head.give_lane_zero (full hidden property) ();
}

ghost
fn scatter
  (hidden:array2 BF.t Native.hidden_layout) (property:snapshot -> prop) ()
  requires forall+ (i:lane). Head.lane_only i (full hidden property)
  ensures forall+ (i:lane). shared hidden property i
{
  Head.take_lane_zero (full hidden property) ();
  unfold full hidden property;
  with values. assert hidden |-> values;
  T.tensor_share_n hidden 32;
  Every.forevery_map #lane
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) values)
    (shared hidden property)
    fn i {
      fold shared hidden property i;
    };
}

// The predicate and pointers are warp-uniform; owned-cell agreement unifies the existential observations.
inline_for_extraction noextract
let gather_wait (hidden:array2 BF.t Native.hidden_layout) (property:snapshot -> prop)
  (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** shared hidden property tid)
      (ensures fun _ -> thread_id 32 tid ** Head.lane_only tid (full hidden property)) =
  Warp.warp_barrier_wait () (shared hidden property)
    (fun i -> Head.lane_only i (full hidden property))
    (gather hidden property ()) #32 #tid

inline_for_extraction noextract
let scatter_wait (hidden:array2 BF.t Native.hidden_layout) (property:snapshot -> prop)
  (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** Head.lane_only tid (full hidden property))
      (ensures fun _ -> thread_id 32 tid ** shared hidden property tid) =
  Warp.warp_barrier_wait ()
    (fun i -> Head.lane_only i (full hidden property))
    (shared hidden property) (scatter hidden property ()) #32 #tid
