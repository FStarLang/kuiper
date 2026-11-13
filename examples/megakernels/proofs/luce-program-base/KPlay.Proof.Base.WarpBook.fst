module KPlay.Proof.Base.WarpBook

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery
open FStar.Ghost

module A = Pulse.Lib.Array
module T = Kuiper.Tensor
module GR = Pulse.Lib.GhostReference
module Every = Kuiper.ForEvery
module Warp = Kuiper.Barrier.Warp
module BF = Kuiper.BFloat16
module Native = KPlay.Proof.Base.NativeProjection
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.CacheNative
module Head = KPlay.Proof.Base.Head
module Hidden = KPlay.Proof.Base.WarpHidden
module CacheShare = KPlay.Proof.Base.WarpCache
module GhostShare = KPlay.Proof.Base.GhostWarp

let lane = Head.lane

inline_for_extraction noextract
let full (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop)
  (valid:a -> Hidden.snapshot -> Cache.snapshot -> prop) : slprop =
  exists* (state:a) (h:Hidden.snapshot) (k:Cache.snapshot).
    GR.pts_to book state ** hidden |-> h ** A.pts_to_mask cache k Init.all_cells **
    local_owners state ** pure (valid state h k)

inline_for_extraction noextract
let chunk (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (state:a) (h:Hidden.snapshot) (k:Cache.snapshot) : slprop =
  GR.pts_to book #(1.0R /. 32) state **
  hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells

inline_for_extraction noextract
let shared (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop) (valid:a -> Hidden.snapshot -> Cache.snapshot -> prop) (i:lane) : slprop =
  exists* (state:a) (h:Hidden.snapshot) (k:Cache.snapshot).
    chunk book hidden cache state h k ** Head.lane_only i (local_owners state) ** pure (valid state h k)

inline_for_extraction noextract
let anchor (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (state:a) (h:Hidden.snapshot) (k:Cache.snapshot) : slprop =
  chunk book hidden cache state h k

ghost
fn scatter
  (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop) (valid:a -> Hidden.snapshot -> Cache.snapshot -> prop) ()
  requires forall+ (i:lane). Head.lane_only i (full book hidden cache local_owners valid)
  ensures forall+ (i:lane). shared book hidden cache local_owners valid i
{
  Head.take_lane_zero (full book hidden cache local_owners valid) ();
  unfold full book hidden cache local_owners valid;
  with state h k. assert GR.pts_to book state ** hidden |-> h ** A.pts_to_mask cache k Init.all_cells;
  GhostShare.share32 book #state #1.0R;
  T.tensor_share_n hidden 32;
  CacheShare.share cache #k #1.0R 32;
  Every.forevery_zip3 #lane
    (fun (_:lane) -> GR.pts_to book #(1.0R /. 32) state)
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) h)
    (fun (_:lane) -> A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells);
  Head.give_lane_zero (local_owners state) ();
  Every.forevery_zip #lane
    (fun (_:lane) -> GR.pts_to book #(1.0R /. 32) state **
      hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells)
    (fun i -> Head.lane_only i (local_owners state));
  Every.forevery_map #lane
    (fun i -> (GR.pts_to book #(1.0R /. 32) state **
      hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells) **
      Head.lane_only i (local_owners state))
    (shared book hidden cache local_owners valid)
    fn i {
      fold chunk book hidden cache state h k;
      fold shared book hidden cache local_owners valid i;
    };
}

ghost
fn gather
  (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop) (valid:a -> Hidden.snapshot -> Cache.snapshot -> prop) ()
  requires forall+ (i:lane). shared book hidden cache local_owners valid i
  ensures forall+ (i:lane). Head.lane_only i (full book hidden cache local_owners valid)
{
  Every.forevery_extract_if_eqtype #lane (0 <: lane) (shared book hidden cache local_owners valid);
  unfold shared book hidden cache local_owners valid (0 <: lane);
  with state h k. assert chunk book hidden cache state h k;
  rewrite Head.lane_only (0 <: lane) (local_owners state) as local_owners state;
  fold anchor book hidden cache state h k;
  Every.forevery_ext #lane
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else shared book hidden cache local_owners valid i)
    (fun i -> Hidden.nonzero i (shared book hidden cache local_owners valid i));
  Every.forevery_map_extra #lane (anchor book hidden cache state h k)
    (fun i -> Hidden.nonzero i (shared book hidden cache local_owners valid i))
    (fun i -> Hidden.nonzero i (chunk book hidden cache state h k))
    fn i {
      if (( = ) #lane i (0 <: lane)) {
        rewrite Hidden.nonzero i (shared book hidden cache local_owners valid i)
          as Hidden.nonzero i (chunk book hidden cache state h k);
      } else {
        rewrite Hidden.nonzero i (shared book hidden cache local_owners valid i) as shared book hidden cache local_owners valid i;
        unfold shared book hidden cache local_owners valid i;
        with other oh ok. assert chunk book hidden cache other oh ok;
        rewrite Head.lane_only i (local_owners other) as emp;
        unfold chunk book hidden cache other oh ok;
        unfold anchor book hidden cache state h k;
        unfold chunk book hidden cache state h k;
        GR.pts_to_injective_eq #a #(1.0R /. 32) #(1.0R /. 32) #state #other book;
        Hidden.agreement hidden #h #oh #(1.0R /. 32) #(1.0R /. 32);
        CacheShare.agreement cache #k #ok #(1.0R /. 32) #(1.0R /. 32);
        rewrite each other as state;
        rewrite each oh as h;
        rewrite each ok as k;
        fold chunk book hidden cache state h k;
        fold anchor book hidden cache state h k;
        fold chunk book hidden cache state h k;
        rewrite chunk book hidden cache state h k as Hidden.nonzero i (chunk book hidden cache state h k);
      }
    };
  Every.forevery_ext #lane
    (fun i -> Hidden.nonzero i (chunk book hidden cache state h k))
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else chunk book hidden cache state h k);
  unfold anchor book hidden cache state h k;
  Every.forevery_unextract_if_eqtype #lane (0 <: lane) (fun _ -> chunk book hidden cache state h k);
  Every.forevery_map #lane
    (fun _ -> chunk book hidden cache state h k)
    (fun _ -> GR.pts_to book #(1.0R /. 32) state **
      hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells)
    fn _ { unfold chunk book hidden cache state h k; };
  Every.forevery_unzip3 #lane
    (fun _ -> GR.pts_to book #(1.0R /. 32) state)
    (fun _ -> hidden |-> Frac (1.0R /. 32) h)
    (fun _ -> A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells);
  GhostShare.gather book #state #1.0R 32;
  T.tensor_gather_n hidden 32;
  CacheShare.gather_same cache #k #1.0R 32;
  fold full book hidden cache local_owners valid;
  Head.give_lane_zero (full book hidden cache local_owners valid) ();
}

inline_for_extraction noextract
let scatter_wait (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop) (valid:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** Head.lane_only tid (full book hidden cache local_owners valid))
      (ensures fun _ -> thread_id 32 tid ** shared book hidden cache local_owners valid tid) =
  Warp.warp_barrier_wait ()
    (fun i -> Head.lane_only i (full book hidden cache local_owners valid))
    (shared book hidden cache local_owners valid)
    (scatter book hidden cache local_owners valid ()) #32 #tid

inline_for_extraction noextract
let gather_wait (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop) (valid:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** shared book hidden cache local_owners valid tid)
      (ensures fun _ -> thread_id 32 tid ** Head.lane_only tid (full book hidden cache local_owners valid)) =
  Warp.warp_barrier_wait () (shared book hidden cache local_owners valid)
    (fun i -> Head.lane_only i (full book hidden cache local_owners valid))
    (gather book hidden cache local_owners valid ()) #32 #tid

ghost
fn strengthen
  (#a:Type0) (book:GR.ref a)
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (local_owners:a -> slprop)
  (pre post:a -> Hidden.snapshot -> Cache.snapshot -> prop) (i:lane)
  (implication:squash (forall (state:a) (h:Hidden.snapshot) (k:Cache.snapshot).
    pre state h k ==> post state h k))
  requires Head.lane_only i (full book hidden cache local_owners pre)
  ensures Head.lane_only i (full book hidden cache local_owners post)
{
  if (( = ) #lane i (0 <: lane)) {
    rewrite Head.lane_only i (full book hidden cache local_owners pre) as full book hidden cache local_owners pre;
    unfold full book hidden cache local_owners pre;
    with state h k. assert GR.pts_to book state ** hidden |-> h ** A.pts_to_mask cache k Init.all_cells;
    fold full book hidden cache local_owners post;
    rewrite full book hidden cache local_owners post as Head.lane_only i (full book hidden cache local_owners post);
  } else {
    rewrite Head.lane_only i (full book hidden cache local_owners pre)
      as Head.lane_only i (full book hidden cache local_owners post);
  }
}

ghost
fn keep (p:lane -> slprop) ()
  preserves forall+ (i:lane). p i
{}

inline_for_extraction noextract
let keep_wait (p:lane -> slprop) (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** p tid)
      (ensures fun _ -> thread_id 32 tid ** p tid) =
  Warp.warp_barrier_wait () p p (keep p ()) #32 #tid

ghost
fn map_lane_zero (i:lane) (pre post:slprop)
  (transform:unit -> stt_ghost unit emp_inames pre (fun _ -> post))
  requires Head.lane_only i pre
  ensures Head.lane_only i post
{
  if (( = ) #lane i (0 <: lane)) {
    rewrite Head.lane_only i pre as pre;
    transform ();
    rewrite post as Head.lane_only i post;
  } else {
    rewrite Head.lane_only i pre as Head.lane_only i post;
  }
}
